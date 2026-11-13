"""OpenVLA-OFT policy server (runs in /content/oft-venv; scripts/colab/setup_oft.sh). D-081.

    python oft_server.py --rlinf /content/RLinf --ckpt /content/ckpt/RLinf-OpenVLAOFT-CALVIN-SFT --port 6901

Mirrors RLinf's CALVIN path for `RLinf/RLinf-OpenVLAOFT-CALVIN-SFT` (tests/e2e_tests/embodied/
calvin_ppo_openvlaoft.yaml with examples/embodiment/config/model/openvla_oft.yaml): the model is built by RLinf's
openvla_oft.rlinf.get_model() and acts through OpenVLAOFTForRLActionPrediction.predict_action_batch with the
env_obs that calvin_gym_env._wrap_obs makes (rgb_static as main image, robot_obs[:7] as state, which the model
does not use) and the eval sampling parameters (temperature 0.6, top-k off). One 7-D relative action per query
(num_action_chunks 1, unnorm_key calvin_abc). The gripper is binarized client-side, > 0 -> +1, as
prepare_actions_for_calvin does. Deliberate difference: torch's RNG is seeded from noise_key before each query
(D-023). rlinf.utils.utils (which imports the FSDP / Ray stack) is replaced by the two functions the model uses;
rlinf's package __init__s are not run.
"""

import argparse
import hashlib
import os
import sys
import threading
import traceback
import types
from multiprocessing.connection import Listener

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CFG = dict(  # model/openvla_oft.yaml defaults, then the CALVIN test config's overrides
    model_type="openvla_oft", implement_version="rlinf", precision="bf16", value_type="action_level", action_dim=7,
    num_action_chunks=1, use_proprio=False, use_film=False, unnorm_key="calvin_abc", center_crop=True,
    max_prompt_length=50, vocab_size=32000, hidden_size=4096, add_value_head=False, image_size=[224, 224],
    is_lora=False, lora_rank=32, lora_path=None, num_images_in_input=1, attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True, trust_remote_code=True, policy_setup="widowx_bridge")
SAMPLING = dict(do_sample=True, temperature=0.6, top_k=-1, top_p=1.0)  # rollout.sampling_params, eval


def seed_from_key(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(key)).encode()).digest()[:8], "little") & (2**63 - 1)


def compute_logprobs_from_logits(logits, target, op_type="torch"):  # rlinf.utils.utils, torch branch
    logprobs = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    return logprobs.view(*logits.shape[:-1]).float()


def compute_entropy_from_logits(logits, dim=-1):  # rlinf.utils.utils
    logp = F.log_softmax(logits, dim=dim)
    return -(logp.exp() * logp).sum(dim=dim)


def load(rlinf, ckpt, device="cuda"):
    import importlib.util

    from omegaconf import OmegaConf

    # prismatic's package __init__s import the training stack (TF data loaders, draccus); register the packages
    # without running them, so only the HF modules and vla.constants / training.train_utils load.
    root = importlib.util.find_spec("prismatic").submodule_search_locations[0]
    for name in ("prismatic", "prismatic.extern", "prismatic.extern.hf", "prismatic.training", "prismatic.vla"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(root, *name.split(".")[1:])]
        sys.modules[name] = pkg
    for name in ("rlinf", "rlinf.models", "rlinf.models.embodiment", "rlinf.models.embodiment.modules",
                 "rlinf.models.embodiment.prismatic", "rlinf.models.embodiment.openvla_oft"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(rlinf, *name.split("."))]
        sys.modules[name] = pkg
    utils = types.ModuleType("rlinf.utils.utils")
    utils.compute_logprobs_from_logits = compute_logprobs_from_logits
    utils.compute_entropy_from_logits = compute_entropy_from_logits
    sys.modules["rlinf.utils"] = types.ModuleType("rlinf.utils")
    sys.modules["rlinf.utils.utils"] = utils
    from rlinf.models.embodiment.openvla_oft.rlinf import get_model

    model = get_model(OmegaConf.create({**CFG, "model_path": ckpt}), torch_dtype=torch.bfloat16)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlinf", default="/content/RLinf")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6901)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true", help="one query on a blank frame, then exit")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = load(a.rlinf, a.ckpt, a.device)
    print(f"OpenVLA-OFT ready on port {a.port}", flush=True)
    lock = threading.Lock()

    def infer(msg):
        env_obs = {  # calvin_gym_env._wrap_obs, batch of one
            "main_images": torch.as_tensor(np.asarray(msg["rgb"]["rgb_static"], dtype=np.uint8))[None],
            "wrist_images": torch.as_tensor(np.asarray(msg["rgb"]["rgb_gripper"], dtype=np.uint8))[None],
            "states": torch.as_tensor(np.asarray(msg["robot_obs"])[:7])[None],
            "task_descriptions": [str(msg["lang"])],
        }
        torch.manual_seed(seed_from_key(msg["noise_key"]))
        with torch.no_grad():
            actions, _ = model.predict_action_batch(env_obs=env_obs, calculate_logprobs=False,
                                                    calculate_values=False, **SAMPLING)
        return np.asarray(actions, dtype=np.float64).reshape(-1, 7)  # (1, 7) relative

    if a.selftest:
        blank = np.zeros((200, 200, 3), np.uint8)
        print("selftest action", infer({"rgb": {"rgb_static": blank, "rgb_gripper": blank[:84, :84]},
                                        "robot_obs": np.zeros(15), "lang": "open the drawer", "noise_key": [0, "t", 0]}))
        return

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

    with Listener(("127.0.0.1", a.port), authkey=b"calvin-max", backlog=64) as listener:
        while True:
            threading.Thread(target=serve, args=(listener.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
