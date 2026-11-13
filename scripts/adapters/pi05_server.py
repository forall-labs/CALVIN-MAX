"""pi0.5 policy server (runs in /content/pi05-venv; scripts/colab/setup_pi05.sh). D-071.

    python pi05_server.py --ckpt /content/ckpt/RLinf-Pi05-CALVIN-ABC-D-SFT --port 6301

Mirrors RLinf toolkits/standalone_eval_scripts/openpi (calvin_eval.py + create_trained_policy):
- inputs: rgb_static (200x200) as base image, rgb_gripper (84x84) as wrist image, state = robot_obs[:3], [3:6],
  [6:7] (ee pos, ee euler, gripper width; RLinf's eval and gym env both pass robot_obs[:7]), prompt = the
  first validation annotation of the subtask;
- transforms: CalvinInputs -> Normalize(norm stats from the checkpoint) -> openpi model transforms, and back;
- sample_kwargs num_steps=5 (RLinf's eval default); the chunk has 5 relative actions, all executed (H=Q=5);
  the gripper is binarized by the client adapter, as in RLinf's eval.
Deliberate differences: (D-023) torch's RNG is seeded from noise_key before each query, so the flow sampling
noise is reproducible and a null branch reproduces Base; sample_actions runs eagerly (openpi torch.compiles it
with CUDA graphs, which break under per-query seeding).
"""

import argparse
import hashlib
import os
import pathlib
import sys
import threading
import traceback
import types
from multiprocessing.connection import Listener

import numpy as np

# Requests are served from one thread per connection, and each thread gets its own cuBLAS handle; without a fixed
# workspace config cuBLAS may reduce in a different order, and one null branch in 38 diverged late (D-073).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

NUM_STEPS = 5


def seed_from_key(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(key)).encode()).digest()[:8], "little") & (2**63 - 1)


def load_policy(ckpt: str):
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms
    import safetensors.torch
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import checkpoints as _checkpoints

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rlinf_calvin import PI05_CALVIN as cfg

    model = pi0_pytorch.PI0Pytorch(config=cfg.model)
    # PI0Pytorch wraps sample_actions in torch.compile(mode="max-autotune"), i.e. CUDA graphs, whose replay fails
    # once torch's RNG is re-seeded per query (and whose graph trees are per thread). Use the eager method.
    model.sample_actions = types.MethodType(pi0_pytorch.PI0Pytorch.sample_actions, model)
    missing, unexpected = safetensors.torch.load_model(model, f"{ckpt}/model.safetensors", strict=False)
    print(f"weights: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(ckpt), data_config.asset_id)
    return _policy.Policy(
        model,
        transforms=[transforms.InjectDefaultPrompt(None), *data_config.data_transforms.inputs,
                    transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                    *data_config.model_transforms.inputs],
        output_transforms=[*data_config.model_transforms.outputs,
                           transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                           *data_config.data_transforms.outputs],
        sample_kwargs={"num_steps": NUM_STEPS},
        metadata=cfg.policy_metadata,
        is_pytorch=True,
        pytorch_device="cuda",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6301)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy = load_policy(args.ckpt)
    print(f"pi0.5 ready on port {args.port} (num_steps={NUM_STEPS})", flush=True)
    lock = threading.Lock()

    def infer(msg):
        ro = np.asarray(msg["robot_obs"], dtype=np.float64)
        element = {
            "observation/image": np.asarray(msg["rgb"]["rgb_static"]),
            "observation/wrist_image": np.asarray(msg["rgb"]["rgb_gripper"]),
            "observation/state_ee_pos": ro[:3], "observation/state_ee_rot": ro[3:6], "observation/state_gripper": ro[6:7],
            "prompt": str(msg["lang"]),
        }
        torch.manual_seed(seed_from_key(msg["noise_key"]))
        return np.asarray(policy.infer(element)["actions"], dtype=np.float64)

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
