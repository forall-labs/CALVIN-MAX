"""FLOWER policy server for CALVIN-MAX (runs in the FLOWER venv; see scripts/colab/setup_flower.sh).

    python flower_server.py --ckpt /content/ckpt/flower_calvin_abc/model.safetensors --port 6100

Mirrors FLOWER's own CALVIN evaluation:
- preprocessing as in HulcWrapper/process_rgb with the checkpoint's `val` transforms: uint8 HWC to CHW,
  Resize(224, antialias), /255, CLIP mean/std;
- `FLOWERVLA.forward`: encode_observations, then sample_actions(noise, inference=True) with 4 Euler steps.

The one deliberate difference: noise comes from a generator seeded by the query's noise_key
(policy_seed, chain_id, global_step) instead of the global RNG. This keeps Base/branch pairing exact (D-023).
The returned chunk is the full act_window_size (10) actions in [-1, 1].
"""

from __future__ import annotations

import os

os.environ["MPLBACKEND"] = "Agg"  # Colab exports its inline backend, which a plain process cannot load

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys
import threading
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def patch_florence_imports():
    """Florence-2's remote code lists flash_attn as an import; it is optional at runtime."""
    from transformers import dynamic_module_utils

    orig = dynamic_module_utils.get_imports

    def get_imports(filename):
        return [i for i in orig(filename) if i != "flash_attn"]

    dynamic_module_utils.get_imports = get_imports


def load_model(ckpt: Path, flower_root: Path, device):
    """Same steps as flower.evaluation.utils.load_mode_from_safetensor (HF format). That module is not
    imported because it pulls in pyhash and calvin_env at import time."""
    import hydra
    from omegaconf import OmegaConf

    sys.path.insert(0, str(flower_root))
    patch_florence_imports()
    config = OmegaConf.load(ckpt.parent / "config.yaml")
    load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), "optimizer": None, "num_sampling_steps": 4})
    if "ckpt_path" in load_cfg:
        del load_cfg["ckpt_path"]
    load_cfg["pretrained_model_path"] = str(ckpt)
    model = hydra.utils.instantiate(load_cfg)
    model.freeze()
    return model.to(device).eval()


def make_preprocess(device):
    tf = T.Compose([T.Resize(224, antialias=True), lambda x: x.float().div(255), T.Normalize(MEAN, STD)])

    def pre(rgb: np.ndarray) -> torch.Tensor:  # (H, W, 3) uint8 -> (1, 1, 3, 224, 224)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).byte().permute(2, 0, 1).unsqueeze(0).to(device)
        return tf(x).unsqueeze(0)

    return pre


def seed_from_key(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(key)).encode()).digest()[:8], "little") & (2**63 - 1)


@torch.no_grad()
def infer(model, pre, msg, device):
    batch = {
        "rgb_obs": {"rgb_static": pre(msg["rgb"]["rgb_static"]), "rgb_gripper": pre(msg["rgb"]["rgb_gripper"])},
        "lang_text": [msg["lang"]],
    }
    features = model.encode_observations(batch)
    gen = torch.Generator(device=device).manual_seed(seed_from_key(msg["noise_key"]))
    noise = torch.randn(len(features["features"]), model.act_window_size, model.action_dim,
                        device=features["features"].device, generator=gen)
    actions = model.sample_actions(noise, features, inference=True)  # (1, 10, 7), clamped to [-1, 1]
    return actions[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--flower-root", type=Path, default=Path("/content/flower_vla_calvin"))
    ap.add_argument("--port", type=int, default=6100)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.ckpt, args.flower_root, device)
    pre = make_preprocess(device)
    print(f"FLOWER ready on port {args.port}", flush=True)

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
                        with lock:  # one GPU inference at a time; simulator workers run in parallel
                            out = infer(model, pre, msg, device)
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
