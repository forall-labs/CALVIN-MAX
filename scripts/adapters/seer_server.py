"""Seer policy server (runs in /content/seer-venv; scripts/colab/setup_seer.sh).

    python seer_server.py --repo /content/Seer --ckpt /content/ckpt/seer/<run>.pth \
        --vit /content/ckpt/seer/mae_pretrain_vit_base.pth --port 6501 [--large]

Mirrors OpenRobotLab/Seer utils/eval_utils_calvin.py (ModelWrapper.step) with the model built as in eval_calvin.py
and scripts/CALVIN_ABC_D/Seer[-Large]/eval.sh. As for GR-1, the 10-frame observation window arrives with each
query (the client's HistoryAdapter keeps it as policy state, cleared at subtask start; frames are keyed, pixels
come once per connection and their preprocessed tensors are cached):
- per frame: rgb_static and rgb_gripper through CLIP's image preprocess; state = robot_obs[:6] + robot_obs[-1];
- a window shorter than 10 is padded by repeating its last frame; the instruction token is repeated 10 times;
- the action is the first of the action_pred_steps predicted at the last real frame: 6 relative arm values +
  gripper > 0.5 -> +1 / -1, cast through float16 as the reference does (one action per step, H = Q = 1).
Seer samples nothing, so noise_key is unused. Deliberate differences, all equal to the reference up to float
rounding: the encoder's random patch shuffle at mask ratio 0 is dropped (it drew from the CUDA RNG that server
threads share, so Base would not reproduce bit for bit); each frame's encoder features are computed once and
cached, the repeated instruction is encoded once, and the unused future-image decoder is skipped. The checkpoint was saved from a DDP model ("module." keys).
"""

import argparse
import sys
import threading
import traceback
from multiprocessing.connection import Listener

import numpy as np
import torch
from PIL import Image

# scripts/CALVIN_ABC_D/Seer/eval.sh (Seer-Large: Seer-Large/eval.sh)
FLAGS = ["--traj_cons", "--rgb_pad", "10", "--gripper_pad", "4", "--bf16_module", "vision_encoder",
         "--precision", "fp32", "--transformer_layers", "24", "--phase", "evaluate", "--finetune_type", "calvin",
         "--action_pred_steps", "3", "--sequence_length", "10", "--future_steps", "3", "--window_size", "13",
         "--obs_pred"]
SIZE = {False: ["--hidden_dim", "384", "--transformer_heads", "12", "--num_resampler_query", "6",
                "--num_obs_token_per_image", "9"],
        True: ["--hidden_dim", "1024", "--transformer_heads", "16", "--num_resampler_query", "16",
               "--num_obs_token_per_image", "16"]}


class CachedEncoder(torch.nn.Module):
    """Seer's MAE encoder with per-frame features cached per connection (as gr1_server.CachedMAE): each frame is
    encoded once, as a batch of one, instead of the whole window every step. `keys` holds one key list per call
    (primary camera, then wrist)."""

    def __init__(self, enc):
        super().__init__()
        self.enc, self.keys, self.cache = enc, [], {}
        enc.random_masking = unshuffled

    def forward_encoder(self, x, mask_ratio):
        out = []
        for i, key in enumerate(self.keys.pop(0)):
            if key not in self.cache:
                self.cache[key] = self.enc.forward_encoder(x[i:i + 1], mask_ratio)[0]
            out.append(self.cache[key])
        return torch.cat(out), None, None


def unshuffled(x, mask_ratio):
    """random_masking at mask_ratio 0 keeps every patch but shuffles them with the global CUDA RNG, which server
    threads share; the result is order-invariant only up to float rounding, so keep the patch order instead."""
    assert mask_ratio == 0
    n, length, _ = x.shape
    return x, torch.zeros(n, length, device=x.device), torch.arange(length, device=x.device).expand(n, length)


def load_policy(repo, ckpt, vit, large, device):
    sys.path.insert(0, repo)
    import clip
    from models.seer_model import SeerAgent
    from utils.arguments_utils import get_parser

    argv = FLAGS + SIZE[large] + ["--vit_checkpoint_path", vit, "--save_checkpoint_path", "checkpoints/"]
    saved, sys.argv = sys.argv, [sys.argv[0], *argv]  # get_parser() itself parses sys.argv before returning
    try:
        args = get_parser(is_eval=True).parse_args(argv)
    finally:
        sys.argv = saved
    model = SeerAgent(
        finetune_type=args.finetune_type, clip_device=device, vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length, num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image, calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size, action_pred_steps=args.action_pred_steps, obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs, attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal, atten_goal_state=args.atten_goal_state, mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers, hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads, phase=args.phase, gripper_width=args.gripper_width)
    model = model.float()
    model.vision_encoder.bfloat16()
    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model = model.to(device)
    model._init_model_type()
    sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, False)
    print(f"weights: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys", flush=True)
    model.eval()
    # The future-image decoder (2 blocks over 20 x 212 tokens, ~1/3 of a query) only feeds the training loss;
    # nothing on the action path reads its output, so skipping it leaves the actions bit for bit unchanged.
    model.image_decoder = torch.nn.Identity()
    model.vision_encoder = CachedEncoder(model.vision_encoder)
    encode_text, text_cache = model.clip_model.encode_text, {}

    def encode_once(tokens):  # the instruction is repeated per window slot; encode it once
        assert bool((tokens == tokens[:1]).all())
        key = tokens[0].cpu().numpy().tobytes()
        if key not in text_cache:
            text_cache[key] = encode_text(tokens[:1])
        return text_cache[key].expand(len(tokens), -1)

    model.clip_model.encode_text = encode_once
    return model, clip, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/Seer")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vit", required=True)
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--port", type=int, default=6501)
    a = ap.parse_args()
    device = "cuda"
    model, clip, args = load_policy(a.repo, a.ckpt, a.vit, a.large, device)
    L = args.sequence_length
    cast = torch.float32
    print(f"Seer{'-Large' if a.large else ''} ready on port {a.port} (seq_len={L})", flush=True)
    lock = threading.Lock()

    def img(x):
        return model.image_processor(Image.fromarray(np.asarray(x, dtype=np.uint8))).unsqueeze(0).unsqueeze(1).to(cast)

    def infer(msg, cache, feats):
        frames = msg["frames"][-L:]
        n = len(frames)
        for f in frames:  # pixels arrive once per frame and connection; keep the preprocessed tensors
            if "rgb" in f:
                cache[f["key"]] = (img(f["rgb"]["rgb_static"]), img(f["rgb"]["rgb_gripper"]))
        live = {f["key"] for f in frames}
        for key in set(cache) - live:
            del cache[key]
        for key in [k for k in feats if k[1] not in live]:
            del feats[key]
        keys = [f["key"] for f in frames] + [frames[-1]["key"]] * (L - n)  # padding repeats the last frame
        model.vision_encoder.cache = feats
        model.vision_encoder.keys = [[("primary", k) for k in keys], [("wrist", k) for k in keys]]
        with torch.no_grad():
            prim = torch.cat([cache[f["key"]][0] for f in frames], dim=1).to(device)
            wrist = torch.cat([cache[f["key"]][1] for f in frames], dim=1).to(device)
            st = torch.from_numpy(np.stack([np.asarray(f["robot_obs"]) for f in frames]))[None].to(cast)
            st = torch.cat([st[..., :6], st[..., [-1]]], dim=-1).to(device)
            text = clip.tokenize([str(msg["lang"])], truncate=True).unsqueeze(1).to(device)
            text = torch.cat([text] * L, dim=1)
            if n < L:
                prim = torch.cat([prim, prim[:, -1:].repeat(1, L - n, 1, 1, 1)], dim=1)
                wrist = torch.cat([wrist, wrist[:, -1:].repeat(1, L - n, 1, 1, 1)], dim=1)
                st = torch.cat([st, st[:, -1:].repeat(1, L - n, 1)], dim=1)
            arm, grip, *_ = model(image_primary=prim, image_wrist=wrist, state=st, text_token=text,
                                  action=torch.zeros(1, L, 7).to(device))
            act = torch.concat((arm[0, :, 0, :], grip[0, :, 0, :] > 0.5), dim=-1)
            act[:, -1] = (act[:, -1] - 0.5) * 2
            act = act.cpu().detach().to(dtype=torch.float16).numpy()
        return act[n - 1 if n < L else -1].astype(np.float64)[None]  # (1, 7)

    def serve(conn):
        cache = {}  # frame key -> preprocessed (static, gripper), this connection only
        feats = {}  # (camera, frame key) -> encoder features, this connection only
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                try:
                    if msg["op"] == "infer":
                        with lock:
                            out = infer(msg, cache, feats)
                        conn.send({"actions": out})
                    elif msg["op"] == "ping":
                        conn.send({"ok": True})
                except Exception:
                    conn.send({"error": traceback.format_exc()})
        finally:
            conn.close()

    with Listener(("127.0.0.1", a.port), authkey=b"calvin-max", backlog=64) as listener:
        while True:
            threading.Thread(target=serve, args=(listener.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
