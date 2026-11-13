"""Export the data behind the project page's policy explorer (Figure 4).

    python website/tools/explorer_data.py > /tmp/explorer.json

Reads results/analysis/maxA_q9_heldout.json (held-out Max-A cases) and prints, per policy, Base / Pre / Mid
success on the affected instruction for each of the eight core changes (percent, with the number of cases),
and the per-channel timing effect Mid - Pre with its 95% chain-bootstrap interval. The page embeds this JSON
inline, in the EXPLORER constant, so it also works when opened as a file.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["FLOWER", "X-VLA", "pi0.5", "UniVLA", "OpenVLA-OFT", "MoDE", "3DDA", "GR-1", "Seer", "RoboFlamingo"]
DISPLAY = {"pi0.5": "π0.5", "3DDA": "3D Diffuser Actor", "Seer": "Seer-Large"}
FAMILY = {"FLOWER": "VLA", "X-VLA": "VLA", "pi0.5": "VLA", "UniVLA": "VLA", "OpenVLA-OFT": "VLA", "MoDE": "Diffusion",
          "3DDA": "Diffusion", "GR-1": "Predictive", "Seer": "Predictive", "RoboFlamingo": "VLM+LSTM"}
CHANGES = ["camera_shift", "sensor_noise_onset", "illumination_switch", "visual_theme_switch", "distractor_burst",
           "target_relocation", "receptacle_relocation", "obstacle_insertion"]


def pct(x):
    return round(100 * x, 1)


def main():
    d = json.loads((ROOT / "results/analysis/maxA_q9_heldout.json").read_text())
    out = {"changes": CHANGES, "policies": []}
    for p in ORDER:
        core, timing = d[p]["core"], d[p]["channel_timing"]
        rows = [[pct(core["core/" + c][k]) for k in ("base_sr", "pre_sr", "mid_sr")] + [core["core/" + c]["n"]]
                for c in CHANGES]
        out["policies"].append({"key": p, "name": DISPLAY.get(p, p), "family": FAMILY[p], "rows": rows,
                                "timing": {k: [pct(v["estimate"])] + [pct(x) for x in v["ci95"]] for k, v in timing.items()}})
    out["pooled_timing"] = {k: [pct(v["estimate"])] + [pct(x) for x in v["ci95"]] for k, v in d["pooled"].items()}
    print(json.dumps(out, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
