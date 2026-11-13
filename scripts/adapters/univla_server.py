"""UniVLA policy server (runs in /content/univla-venv; scripts/colab/setup_univla.sh).

    python univla_server.py --repo /content/UniVLA --ckpt /content/ckpt/univla-7b-224-sft-calvin --port 6601

Mirrors OpenDriveLab/UniVLA experiments/robot/calvin (calvin_model.py WrappedCalvinEvaluation.step,
run_calvin_eval_ddp.py rollout, openvla_utils.get_vla_latent_action):
- input: rgb_static only, prompt "In: What action should the robot take to {instruction.lower()}?\nOut:"
  (no history action: the eval resets it to '' and never sets it);
- the VLA generates 4 latent-action tokens (sampling, temperature 0.75, top-p 0.9); their last-layer hidden
  states and the 256 visual embeddings go through the action decoder, which returns a chunk of `window` relative
  actions, all executed (H = Q = window); the gripper is binarized client-side by sign, as in the eval.
The policy is stateless (the decoder's reset() is a no-op). Deliberate differences: torch's RNG is seeded from
noise_key before each query (D-023), so a null branch reproduces Base; attention runs as SDPA instead of
flash-attention-2 (no flash-attn build); the reproduction check covers both.
ActionDecoder is copied from calvin_model.py (Apache-2.0, OpenDriveLab) so the eval stack, which imports
TensorFlow and calvin_agent, is not needed; its parameter names match action_decoder.pt.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import traceback
from multiprocessing.connection import Listener

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # one cuBLAS handle per server thread (D-073)
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def seed_from_key(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(key)).encode()).digest()[:8], "little") & (2**63 - 1)


def load(repo, ckpt, device):
    import types

    # prismatic/__init__.py and prismatic/models/__init__.py import the whole training stack (draccus, TF data
    # loaders, ...). Register the packages without running them, so only the HF inference modules load.
    for name in ("prismatic", "prismatic.models"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(repo, *name.split("."))]
        sys.modules[name] = pkg
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.policy.transformer_utils import MAPBlock
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    class ActionDecoder(torch.nn.Module):
        def __init__(self, window_size=5, hidden_dim=512):
            super().__init__()
            self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
            self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
            self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 7 * window_size), torch.nn.Tanh())

        def forward(self, latent_action_tokens, visual_embed):
            latent_action_tokens = latent_action_tokens[:, -4:]
            visual_embed = self.visual_pool(visual_embed)
            return self.proj(self.latent_action_pool(latent_action_tokens, init_embed=visual_embed))

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    vla = AutoModelForVision2Seq.from_pretrained(ckpt, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, trust_remote_code=True).to(device)
    vla.eval()
    vla.norm_stats = json.load(open(f"{ckpt}/dataset_statistics.json"))
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    sd = torch.load(f"{ckpt}/action_decoder.pt", map_location="cpu")
    window = sd["proj.0.weight"].shape[0] // 7
    decoder = ActionDecoder(window)
    decoder.load_state_dict(sd)
    decoder.to(device).eval()
    return vla, processor, decoder, window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/UniVLA")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6601)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda"
    vla, processor, decoder, window = load(args.repo, args.ckpt, device)
    print(f"UniVLA ready on port {args.port} (action chunk {window})", flush=True)
    lock = threading.Lock()

    def infer(msg):
        image = Image.fromarray(np.asarray(msg["rgb"]["rgb_static"], dtype=np.uint8)).convert("RGB")
        prompt = f"In: What action should the robot take to {str(msg['lang']).lower()}?\nOut:"
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        torch.manual_seed(seed_from_key(msg["noise_key"]))
        with torch.no_grad():
            latent, visual, _ = vla.predict_latent_action(**inputs, unnorm_key="calvin", do_sample=True,
                                                          temperature=0.75, top_p=0.9)
            action = decoder(latent.to(torch.float), visual.to(torch.float)).reshape(-1, window, 7)
        return np.array(action.tolist(), dtype=np.float64)[0]  # (window, 7) relative actions

    def serve(conn):
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                try:
                    if msg["op"] == "infer":
                        with lock:
                            out = infer(msg)
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
