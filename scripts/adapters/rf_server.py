"""RoboFlamingo policy server (runs in /content/rf-venv; scripts/colab/setup_rf.sh).

    python rf_server.py --repo /content/RoboFlamingo \
        --ckpt /content/ckpt/rf/ABC_D/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth \
        --lm /content/ckpt/rf/mpt-1b-redpajama-200b-dolly \
        --of /content/ckpt/rf/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt --port 6701

Checkpoint: robovlms/RoboFlamingo ABC_D, MPT-IFT-3B ("dolly"), the backbone of the paper's main table (ABCD->D 4.09
equals the README's MPT-IFT-3B row), so it is taken to be the one behind the published ABC->D 2.48.

The model is built by RoboFlamingo's own robot_flamingo/eval/eval_calvin.py main() with the flags of
robot_flamingo/pt_eval_ckpts.bash for this checkpoint (use_gripper, no state, fusion post, window 12; LSTM head),
and it acts through the reference ModelWrapper.step, one wrapper per connection. main() is stopped where it would
start the CALVIN loop, and the simulator-side imports it does not use are stubbed.

State. Each step runs the full Flamingo forward on one frame (pad_length -1: no KV cache, no feature cache), so a
frame's features depend only on that frame. Everything carried across steps is in the LSTM head
(action_head.py): for the first 12 frames of a subtask the hidden state is carried step by step; from the 13th on,
the last 12 frame features are re-run from a zero hidden state. The client's HistoryAdapter keeps the last 13
frames (`--adapter history:13`), so after a restore the server replays the reference's own per-step calls over the
window. With up to 12 frames that is the subtask from its start; with 13, the final step re-runs the last 12, as
Base did. The result is bit for bit what Base computed. In Base, each query adds one frame to what this connection
already holds, and only that frame is stepped. The policy samples nothing, so noise_key is unused.
"""

import argparse
import os
import sys
import threading
import traceback
import types
from multiprocessing.connection import Listener

import numpy as np
import torch


class _Loaded(Exception):
    pass


class _Wrap(torch.nn.Module):
    """Stands in for DistributedDataParallel: same `.module` and "module."-prefixed state dict (the checkpoint was
    saved from DDP), without the gradient buckets DDP allocates (4 GB for the 1.1B trainable parameters). With one
    process and no gradients, DDP's forward is the module's forward."""

    def __init__(self, module, **kw):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _stub(name, **attrs):
    """A module whose every missing attribute is a placeholder class (enough for `from m import a, b`)."""
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    mod.__getattr__ = lambda attr: type(attr, (), {})
    sys.modules[name] = mod
    return mod


def preprocess_image(sample, image_processor):  # robot_flamingo/data/data.py
    return torch.cat([image_processor(s).unsqueeze(0) for s in sample], dim=0)


def preprocess_text_calvin(sample, tokenizer):  # robot_flamingo/data/data.py
    tokenizer.padding_side = "right"
    sample = [f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}" for s in sample]
    text = tokenizer(sample, max_length=32, padding="longest", truncation="only_first", return_tensors="pt")
    return text["input_ids"], text["attention_mask"]


def load(repo, ckpt, lm, of, dist_port):
    sys.path[:0] = [repo, os.path.join(repo, "open_flamingo"), os.path.join(repo, "robot_flamingo", "eval")]
    for name in ("calvin_agent", "calvin_agent.models", "calvin_agent.models.calvin_base_model",
                 "calvin_agent.evaluation", "calvin_agent.evaluation.multistep_sequences",
                 "calvin_agent.evaluation.utils", "calvin_agent.utils", "calvin_agent.utils.utils", "calvin_env",
                 "calvin_env.envs", "calvin_env.envs.play_table_env", "moviepy", "moviepy.editor", "pyrender", "hydra",
                 "pytorch_lightning", "termcolor", "wandb", "robot_flamingo.utils"):
        _stub(name)
    _stub("robot_flamingo.data")
    _stub("robot_flamingo.data.data", preprocess_image=preprocess_image, preprocess_text_calvin=preprocess_text_calvin)
    import eval_calvin as ec
    import eval_utils
    from robot_flamingo.models.factory import mpt_dict

    llm = next(n for n in ["mpt_3b", "mpt_4b", "mpt_9b", "mpt_dolly_3b", "mpt_base_4b"] if n in ckpt)  # as main()
    mpt_dict[llm].update(lang_encoder_path=lm, tokenizer_path=lm, openflamingo_checkpoint=of)
    loaded = {}

    def capture(args, model, image_processor, tokenizer, **kw):
        loaded.update(args=args, model=model, image_processor=image_processor, tokenizer=tokenizer)
        raise _Loaded

    ec.eval_one_epoch_calvin_ddp = capture
    ec.DDP = _Wrap
    window = int(ckpt.split("_ws_")[1].split("_")[0])
    saved, sys.argv = sys.argv, ["eval_calvin.py", "--precision", "fp32", "--use_gripper", "--window_size", str(window),
                                 "--fusion_mode", "post", "--run_name", "RobotFlamingoDBG", "--calvin_dataset", "unused",
                                 "--lm_path", lm, "--tokenizer_path", lm, "--cross_attn_every_n_layers", "4",
                                 "--evaluate_from_checkpoint", ckpt, "--calvin_conf_path", "unused", "--workers", "1",
                                 "--dist-url", f"tcp://127.0.0.1:{dist_port}"]
    try:
        ec.main()
    except _Loaded:
        pass
    finally:
        sys.argv = saved
    a = loaded["args"]
    assert a.head_type == "lstm" and a.pad_length == -1 and not a.use_state and a.fusion_mode == "post", vars(a)

    def wrapper():  # as eval_one_epoch_calvin_ddp builds it
        return eval_utils.ModelWrapper(loaded["model"], loaded["tokenizer"], loaded["image_processor"],
                                       eval_utils.get_cast_dtype(a.precision), False, history_len=None,
                                       future_act_len=a.future_act_len)

    m = loaded["model"].module  # MPTFlamingo forces sep_lm_head: the LSTM head is m.lm_head, lang_encoder's an Identity
    return (m.lm_head if m.sep_lm_head else m.lang_encoder.lm_head), wrapper, a.eval_hist_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/RoboFlamingo")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument("--of", required=True)
    ap.add_argument("--port", type=int, default=6701)
    args = ap.parse_args()
    head, make_wrapper, hist = load(args.repo, args.ckpt, args.lm, args.of, args.port + 1000)
    print(f"RoboFlamingo ready on port {args.port} (LSTM history {hist}, client window {hist + 1})", flush=True)
    lock = threading.Lock()

    class Conn:
        def __init__(self):
            self.wrapper, self.hidden, self.memory, self.keys, self.pixels = make_wrapper(), None, [], [], {}

        def reset(self):
            self.wrapper.reset()
            self.hidden, self.memory, self.keys = None, [], []

        def step(self, f):
            obs = {"rgb_obs": self.pixels[f["key"]], "robot_obs": np.asarray(f["robot_obs"])}
            head.hidden_state, head.history_memory = self.hidden, self.memory
            action = self.wrapper.step(obs, self.lang)
            self.hidden, self.memory = head.hidden_state, head.history_memory
            self.keys.append(f["key"])
            return action

    def infer(msg, c):
        frames = msg["frames"][-(hist + 1):]
        for f in frames:
            if "rgb" in f:
                c.pixels[f["key"]] = f["rgb"]
        live = {f["key"] for f in frames}
        for key in set(c.pixels) - live:
            del c.pixels[key]
        c.lang = str(msg["lang"])
        if len(frames) > 1 and c.keys and c.keys[-1] == frames[-2]["key"]:
            todo = frames[-1:]  # Base: one new frame on top of this connection's history
        else:
            c.reset()  # subtask start, or a restored window: replay it from the subtask's reset
            todo = frames
        for f in todo:
            action = c.step(f)
        return np.asarray(action, dtype=np.float64)[None]  # (1, 7), gripper already +-1

    def serve(conn):
        c = Conn()
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                try:
                    if msg["op"] == "infer":
                        with lock:
                            out = infer(msg, c)
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
