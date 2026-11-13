"""X-VLA policy server (runs in /content/xvla-venv; scripts/colab/setup_xvla.sh).

    python xvla_server.py --model 2toINF/X-VLA-Calvin-ABC_D --port 6201

Mirrors the act() endpoint of the checkpoint's own modeling_xvla.py:
- processor([PIL(static), PIL(gripper)], instruction);
- proprio (1, 20) and domain_id 2 (CALVIN), both from 2toINF/X-VLA evaluation/calvin/calvin_client.py;
- generate_actions(steps=10), returning all 30 actions.
The one deliberate difference (D-023): torch's RNG is seeded from noise_key before each query, which makes
the flow-matching initial noise reproducible.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from multiprocessing.connection import Listener  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

DOMAIN_ID_CALVIN = 2
STEPS = 10


def seed_from_key(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(key)).encode()).digest()[:8], "little") & (2**63 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="2toINF/X-VLA-Calvin-ABC_D")
    ap.add_argument("--port", type=int, default=6201)
    args = ap.parse_args()
    from transformers import AutoModel, AutoProcessor

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    print(f"X-VLA ready on port {args.port} (dtype={dtype}, steps={STEPS})", flush=True)
    lock = threading.Lock()

    def to_model(t):
        t = torch.as_tensor(t)
        return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

    @torch.no_grad()
    def infer(msg):
        images = [Image.fromarray(np.asarray(msg["rgb"]["rgb_static"])), Image.fromarray(np.asarray(msg["rgb"]["rgb_gripper"]))]
        inputs = processor(images, msg["lang"])
        inputs = {k: to_model(v) for k, v in inputs.items()}
        inputs["proprio"] = to_model(torch.as_tensor(np.asarray(msg["proprio"], dtype=np.float32)).unsqueeze(0))
        inputs["domain_id"] = torch.tensor([DOMAIN_ID_CALVIN], dtype=torch.long, device=device)
        torch.manual_seed(seed_from_key(msg["noise_key"]))
        return model.generate_actions(**inputs, steps=STEPS).squeeze(0).float().cpu().numpy()

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
