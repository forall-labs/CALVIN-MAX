"""MoDE policy server for CALVIN-MAX (runs in the policy venv; see scripts/colab/setup_mode.sh).

    python mode_server.py --ckpt-dir /content/ckpt/MoDE_CALVIN_ABC_1 --port 6101

Mirrors MoDE's own CALVIN evaluation (intuitive-robots/MoDE_Diffusion_Policy, mode/evaluation):
- the model comes from the checkpoint's config.yaml with eval override num_sampling_steps=5
  (conf/mode_evaluate.yaml); DDIM sampler from the config;
- preprocessing uses the checkpoint's `val` transforms: static Resize(224), gripper Resize(112), /255, CLIP
  mean/std; shape (1, 1, C, H, W);
- language: raw instruction text through MoDE's CLIP language buffer (goal["lang_text"]);
- forward() returns the full act_window_size (10) action chunk.

The one deliberate difference is the same as for FLOWER (D-023). Before each query, torch's RNG is seeded
from the noise_key, which makes the initial x_T ~ N(0, sigma_max^2) of the DDIM sampler reproducible.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from multiprocessing.connection import Listener  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def load_model(ckpt_dir: Path, mode_root: Path, device):
    """Same steps as mode.evaluation.utils.load_mode_from_safetensor + the eval override."""
    import hydra
    from omegaconf import OmegaConf

    sys.path.insert(0, str(mode_root))
    config = OmegaConf.load(ckpt_dir / "config.yaml")
    load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), "optimizer": None, "num_sampling_steps": 5})
    load_cfg["ckpt_path"] = str(ckpt_dir)
    model = hydra.utils.instantiate(load_cfg)
    model.freeze()
    return model.to(device).eval()


def make_preprocess(device, size):
    tf = T.Compose([T.Resize(size, antialias=True), lambda x: x.float().div(255), T.Normalize(MEAN, STD)])

    def pre(rgb: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(rgb)).byte().permute(2, 0, 1).unsqueeze(0).to(device)
        return tf(x).unsqueeze(0)

    return pre


def seed_from_key(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(key)).encode()).digest()[:8], "little") & (2**63 - 1)


@torch.no_grad()
def infer(model, pre_static, pre_gripper, msg):
    obs = {"rgb_obs": {"rgb_static": pre_static(msg["rgb"]["rgb_static"]),
                       "rgb_gripper": pre_gripper(msg["rgb"]["rgb_gripper"])}}
    torch.manual_seed(seed_from_key(msg["noise_key"]))
    actions = model(obs, {"lang_text": msg["lang"]})  # (1, 10, 7)
    return actions[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--mode-root", type=Path, default=Path("/content/MoDE_Diffusion_Policy"))
    ap.add_argument("--port", type=int, default=6101)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.ckpt_dir, args.mode_root, device)
    pre_s, pre_g = make_preprocess(device, 224), make_preprocess(device, 112)
    print(f"MoDE ready on port {args.port} (steps={model.num_sampling_steps}, sampler={model.sampler_type})", flush=True)
    lock = threading.Lock()

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
                            out = infer(model, pre_s, pre_g, msg)
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
