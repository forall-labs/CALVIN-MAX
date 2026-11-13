"""Does the RoboFlamingo checkpoint load into the model rf_server builds? Prints key overlap (run in rf-venv).

    python scripts/probe/rf_keys.py --repo /content/RoboFlamingo --ckpt ... --lm ... --of ...
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
import rf_server  # noqa: E402

ap = argparse.ArgumentParser()
for a in ("--repo", "--ckpt", "--lm", "--of"):
    ap.add_argument(a, required=True)
args = ap.parse_args()
head, make_wrapper, hist = rf_server.load(args.repo, args.ckpt, args.lm, args.of, 7999)
model = make_wrapper().model
have = model.state_dict()
sd = torch.load(args.ckpt, map_location="cpu")
print("checkpoint top-level keys:", list(sd)[:10])
sd = sd["model_state_dict"]
ck = set(sd)
print(f"checkpoint {len(ck)} keys, model {len(have)} keys, shared {len(ck & set(have))}")
print("checkpoint-only examples:", sorted(ck - set(have))[:8])
print("model-only examples:", sorted(set(have) - ck)[:8])
same = [k for k in ck & set(have) if sd[k].shape == have[k].shape and torch.equal(sd[k].float(), have[k].float().cpu())]
print(f"shared keys whose loaded value equals the checkpoint: {len(same)}")
head_ck = sorted(k for k in ck if "lm_head" in k or "action_head" in k)
head_model = sorted(k for k in have if "lm_head" in k)
print("checkpoint head keys:", len(head_ck), head_ck[:40])
print("model lm_head keys:", len(head_model), [(k, k in ck) for k in head_model][:40])
print("all checkpoint-only:", sorted(ck - set(have)))
