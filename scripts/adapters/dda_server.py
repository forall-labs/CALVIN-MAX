"""3D Diffuser Actor policy server (runs in /content/dda-venv; scripts/colab/setup_dda.sh).

    python dda_server.py --repo /content/3d_diffuser_actor --ckpt /content/ckpt/dda/diffuser_actor_calvin.pth --port 6801

Mirrors nickgkan/3d_diffuser_actor online_evaluation_calvin: the model is built by its create_model() with the
flags of scripts/test_trajectory_calvin.sh, and each query runs what rollout() does before env.step:
- prepare_visual_states: deproject both depth images with the cameras the client sends (utils_with_calvin.deproject),
  rgb / 255, gripper rgb (bilinear) and point cloud (nearest) resized to the static camera's size;
- prepare_proprio_states over the client's history of the last 3 query-time robot_obs: [pos, quaternion of the
  XYZ euler angles (convert_rotation), (gripper + 1) / 2];
- encode_instruction, then DiffusionModel.step under torch.cuda.amp.autocast().
The reply is the whole trajectory (n, 7): absolute [pos, euler XYZ, gripper +-1], all executed by the reference.
Deliberate difference: torch's RNG is seeded from noise_key before each query (D-023), so a null branch reproduces
Base; the reference seeds once per process. Bounds: the CALVIN task_ABC_D statistics.yaml act bounds (same file in
training/ and validation/; extracted from the dataset zip, which we do not otherwise download).
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

ACT_MIN = [-0.432188, -0.545456, 0.293439]  # task_ABC_D/validation/statistics.yaml act_min_bound[:3]
ACT_MAX = [0.42977, 0.139396, 0.796262]  # act_max_bound[:3]
FLAGS = ["--tasks", "A", "B", "C", "D", "--text_encoder", "clip", "--text_max_length", "16", "--backbone", "clip",
         "--gripper_loc_bounds", "tasks/calvin_rel_traj_location_bounds_task_ABC_D.json",
         "--gripper_loc_bounds_buffer", "0.01", "--embedding_dim", "192", "--action_dim", "7", "--use_instruction", "1",
         "--rotation_parametrization", "6D", "--diffusion_timesteps", "25", "--interpolation_length", "20",
         "--num_history", "3", "--relative_action", "1", "--fps_subsampling_factor", "3", "--lang_enhanced", "1",
         "--quaternion_format", "wxyz"]


def seed_from_key(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(key)).encode()).digest()[:8], "little") & (2**63 - 1)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    mod.__getattr__ = lambda attr: type(attr, (), {})
    sys.modules[name] = mod


def load(repo, ckpt):
    os.chdir(repo)  # the flags' relative paths (tasks/...) and the reference's imports are repo-relative
    sys.path.insert(0, repo)
    for name in ("calvin_agent", "calvin_agent.models", "calvin_agent.models.calvin_base_model", "calvin_env",
                 "calvin_env.envs", "calvin_env.envs.play_table_env", "calvin_env.robot", "calvin_env.robot.robot",
                 "calvin_env.utils", "calvin_env.utils.utils", "hydra"):
        _stub(name)
    from online_evaluation_calvin.evaluate_model import create_model
    from online_evaluation_calvin.evaluate_policy import Arguments
    from utils.common_utils import get_gripper_loc_bounds
    from utils.utils_with_calvin import convert_rotation, deproject

    args = Arguments().parse_args(FLAGS + ["--checkpoint", ckpt])
    args.gripper_loc_bounds = get_gripper_loc_bounds(args.gripper_loc_bounds, task=None,  # len(tasks) > 1
                                                     buffer=args.gripper_loc_bounds_buffer)
    args.calvin_gripper_loc_bounds = np.stack([ACT_MIN, ACT_MAX])
    return create_model(args), convert_rotation, deproject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/3d_diffuser_actor")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6801)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    import torch.nn.functional as F

    model, convert_rotation, deproject = load(a.repo, a.ckpt)
    n = model.args.interpolation_length - 1
    print(f"3DDA ready on port {a.port} (trajectory {n})", flush=True)
    lock = threading.Lock()

    def visual(msg):  # evaluate_utils.prepare_visual_states, with the client's cameras
        cams = {k: types.SimpleNamespace(**v) for k, v in msg["cams"].items()}
        ds, dg = (np.asarray(msg["depth"][k], dtype=np.float32) for k in ("depth_static", "depth_gripper"))
        static_pcd = deproject(cams["static"], ds, homogeneous=False, sanity_check=False).transpose(1, 0)
        static_pcd = np.reshape(static_pcd, (ds.shape[0], ds.shape[1], 3))
        gripper_pcd = deproject(cams["gripper"], dg, homogeneous=False, sanity_check=False).transpose(1, 0)
        gripper_pcd = np.reshape(gripper_pcd, (dg.shape[0], dg.shape[1], 3))
        rgb_static = np.asarray(msg["rgb"]["rgb_static"]) / 255.
        rgb_gripper = np.asarray(msg["rgb"]["rgb_gripper"]) / 255.
        h, w = rgb_static.shape[:2]
        rgb_gripper = F.interpolate(torch.as_tensor(rgb_gripper).permute(2, 0, 1).unsqueeze(0), size=(h, w),
                                    mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy()
        gripper_pcd = F.interpolate(torch.as_tensor(gripper_pcd).permute(2, 0, 1).unsqueeze(0), size=(h, w),
                                    mode="nearest").squeeze(0).permute(1, 2, 0).numpy()
        return {"rgb_obs": {"rgb_static": rgb_static, "rgb_gripper": rgb_gripper},
                "pcd_obs": {"pcd_static": static_pcd, "pcd_gripper": gripper_pcd}}

    def proprio(ro):  # evaluate_utils.prepare_proprio_states, one entry
        ro = np.asarray(ro)
        return np.concatenate([ro[:3], convert_rotation(ro[3:6]), (ro[[-1]] + 1) / 2], axis=-1)

    def infer(msg):
        obs = visual(msg)
        obs["proprio"] = np.stack([proprio(ro) for ro in msg["proprio_obs"]], axis=0)
        torch.manual_seed(seed_from_key(msg["noise_key"]))
        lang = model.encode_instruction(str(msg["lang"]), model.args.device)
        with torch.cuda.amp.autocast():
            trajectory = model.step(obs, lang)
        return np.asarray(trajectory, dtype=np.float64)[0]  # (n, 7)

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
