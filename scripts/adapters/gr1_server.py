"""GR-1 policy server (runs in /content/gr1-venv; scripts/colab/setup_gr1.sh).

    python gr1_server.py --repo /content/GR-1 --ckpt /content/ckpt/gr1/snapshot_ABC.pt \
        --mae /content/ckpt/gr1/mae_pretrain_vit_base.pth --port 6401

Mirrors bytedance/GR-1 evaluation/calvin_evaluation.py (GR1CalvinEvaluation.step) exactly, except that the
observation window arrives with each query: the client's HistoryAdapter keeps the last 10 frames of the
subtask as policy state (cleared at subtask start, as model.reset() does), so a branch resumes with Base's
history. Frames are keyed; pixels come once per connection and their preprocessed tensors are cached.
Deliberate difference: each frame's MAE features are computed once, as a batch of one, and cached (CachedMAE);
the reference re-encodes the whole window each step, 10x the encoder work. Same values up to float rounding.
- per frame: rgb_static and rgb_gripper -> ToTensor -> bicubic resize to 224 -> ImageNet normalize;
  state = robot_obs[:6] (tcp pos, euler) and robot_obs[-1] (gripper action, one-hot);
- the window is zero-padded to 10 with an attention mask; the action is the prediction at the last valid
  position: 6 relative arm values + gripper sigmoid > 0.5 -> +1 / -1 (one action per step, H = Q = 1).
GR-1 is deterministic (no sampling), so noise_key is unused.
The reference used torch 1.12 / torchvision 0.13, whose tensor Resize does not antialias; newer torchvision
antialiases by default, so antialias=False is set explicitly.
"""

import argparse
import json
import sys
import threading
import traceback
from multiprocessing.connection import Listener

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class CachedMAE(torch.nn.Module):
    """GR-1's MAE encoder with per-frame features cached per connection. The reference encodes all 10 window
    slots of both cameras at every step; here each frame (and the zero padding frame) is encoded once, always as
    a batch of one, so its features do not depend on the window it arrives in and a branch that re-sends Base's
    window gets Base's features bit for bit. `keys` holds one key list per call (static camera, then gripper)."""

    def __init__(self, mae):
        super().__init__()
        self.mae, self.keys, self.cache = mae, [], {}

    def forward(self, x):
        out = []
        for i, key in enumerate(self.keys.pop(0)):
            if key not in self.cache:
                self.cache[key] = self.mae(x[i:i + 1])
            out.append(self.cache[key])
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


def load_policy(repo, ckpt, mae, device):
    sys.path.insert(0, repo)
    import clip
    import models.vision_transformer as vits
    from models.gr1 import GR1

    v = json.load(open(f"{repo}/logs/configs.json"))
    model_clip, _ = clip.load(v["clip_backbone"], device=device)
    model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
    model_mae.to(device)
    model_mae.load_state_dict(torch.load(mae, map_location="cpu")["model"], strict=False)
    policy = GR1(
        model_clip=model_clip, model_mae=model_mae, state_dim=v["state_dim"], act_dim=v["act_dim"],
        hidden_size=v["embed_dim"], sequence_length=v["seq_len"], training_target=["act_pred"],
        img_feat_dim=v["img_feat_dim"], lang_feat_dim=v["lang_feat_dim"], patch_feat_dim=v["patch_feat_dim"],
        resampler_params=dict(depth=v["resampler_depth"], dim_head=v["resampler_dim_head"], heads=v["resampler_heads"],
                              num_latents=v["resampler_num_latents"], num_media_embeds=v["resampler_num_media_embeds"]),
        without_norm_pix_loss=v["without_norm_pix_loss"], use_hand_rgb=v["use_hand_rgb"], n_layer=v["n_layer"],
        n_head=v["n_head"], n_inner=4 * v["embed_dim"], activation_function=v["activation_function"],
        n_positions=v["n_positions"], resid_pdrop=v["dropout"], attn_pdrop=v["dropout"])
    msg = policy.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    print(f"weights: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys", flush=True)
    policy.to(device).eval()
    policy.model_mae = CachedMAE(policy.model_mae)
    return policy, v, clip.tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/GR-1")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mae", required=True)
    ap.add_argument("--port", type=int, default=6401)
    args = ap.parse_args()
    device = "cuda"
    policy, v, tokenize = load_policy(args.repo, args.ckpt, args.mae, device)
    L, act_dim = v["seq_len"], v["act_dim"]
    preprocess = T.Compose([T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=False),
                            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    print(f"GR-1 ready on port {args.port} (seq_len={L})", flush=True)
    lock = threading.Lock()

    def img(a):
        return preprocess(T.ToTensor()(np.asarray(a, dtype=np.uint8)))

    def infer(msg, cache, feats):
        frames = msg["frames"][-L:]
        n = len(frames)
        for f in frames:  # pixels arrive once per frame and connection; keep the preprocessed tensors
            if "rgb" in f:
                cache[f["key"]] = (img(f["rgb"]["rgb_static"]), img(f["rgb"]["rgb_gripper"]))
        live = {f["key"] for f in frames}
        for key in set(cache) - live:
            del cache[key]
        for key in [k for k in feats if k[1] is not None and k[1] not in live]:
            del feats[key]
        keys = [f["key"] for f in frames] + [None] * (L - n)  # None: the zero padding frame
        policy.model_mae.cache = feats
        policy.model_mae.keys = [[("static", k) for k in keys], [("gripper", k) for k in keys]]
        rgb = torch.stack([cache[f["key"]][0] for f in frames])
        hand = torch.stack([cache[f["key"]][1] for f in frames])
        st = torch.from_numpy(np.stack([np.hstack([f["robot_obs"][:6], f["robot_obs"][-1]]) for f in frames]))
        rgb_data = torch.zeros((1, L, *rgb.shape[1:]))
        rgb_data[0, :n] = rgb
        hand_data = torch.zeros((1, L, *hand.shape[1:]))
        hand_data[0, :n] = hand
        grip = -torch.ones((1, L)).float()
        grip[0, :n] = st[:, 6].float()
        grip = F.one_hot(((grip + 1.0) / 2).long(), num_classes=2).float()
        arm = torch.zeros((1, L, act_dim - 1)).float()
        arm[0, :n] = st[:, :6].float()
        mask = torch.zeros(1, L).long()
        mask[0, :n] = 1
        with torch.no_grad():
            pred = policy(rgb=rgb_data.to(device), hand_rgb=hand_data.to(device),
                          state={"arm": arm.to(device), "gripper": grip.to(device)},
                          language=tokenize(str(msg["lang"])).to(device), attention_mask=mask.to(device))
        valid = mask.flatten() > 0
        a = pred["arm_action_preds"].view(-1, act_dim - 1)[valid.to(device)][-1]
        g = pred["gripper_action_preds"].flatten()[valid.to(device)][-1:]
        g = (torch.sigmoid(g) > 0.5).float() * 2.0 - 1.0
        return torch.cat((a, g)).detach().cpu().numpy().astype(np.float64)[None]  # (1, 7)

    def serve(conn):
        cache = {}  # frame key -> preprocessed (static, gripper), this connection only
        feats = {}  # (camera, frame key) -> MAE features, this connection only
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

    with Listener(("127.0.0.1", args.port), authkey=b"calvin-max", backlog=64) as listener:
        while True:
            threading.Thread(target=serve, args=(listener.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
