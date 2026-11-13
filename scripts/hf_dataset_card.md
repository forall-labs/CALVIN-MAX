---
license: mit
pretty_name: CALVIN-MAX
tags:
- robotics
- benchmark
- calvin
- robustness
- vision-language-action
---

# CALVIN-MAX (work in progress, private)

Mid-execution event benchmark data for CALVIN (ABC→D, environment D). It is compatible with LIBERO-MAX
(arXiv 2609.36518) and adds:
- **Base / Pre / Mid onset-controlled arms**;
- an onset-phase sweep;
- embodiment, intent and physical-delivery events;
- chain-level and privileged-state behaviour measures.

Code: the CALVIN-MAX GitHub repository (private). Design and decisions: `docs/RESEARCH_PLAN.md` and `docs/DECISIONS.md` there.

## Layout
| Path | Content |
|---|---|
| `benchmark/chains/eval_1000.json` | The official 1,000 CALVIN evaluation chains with initial states (pyhash-exact) |
| `benchmark/chains/dev_200.json` | Disjoint dev chains, used only for calibration |
| `benchmark/manifests/max_v*.json`, `lite_v*.json` | Frozen case manifests (policy-independent) and Lite subsets |
| `results/calibration/` | Onset radius calibration (r = 6 cm) |
| `results/base/<policy>/` | Base rollouts: `summary.json` (average length, success at each position), `traces/<chain>.pkl` |
| `results/cases/<policy>_<run>/` | Branch arms: `results.jsonl` (one row per case × arm), `summary.json`, `behaviour.jsonl`, `traces/<case>.<arm>.pkl` |

Traces are pickled `calvin_max.rollout.Trace` objects. Each holds per-step actions (float64), state hashes,
TCP, entity poses and AABBs, contacts, `scene_obs`, subtask boundaries and outcomes. Load them with the
`calvin_max` package from the code repo.

## Policies so far
| Policy | Checkpoint | Base average length (1,000 chains) | Published |
|---|---|---|---|
| FLOWER | `mbreuss/flower_calvin_abc` | 4.531 | 4.53 |
| MoDE | `mbreuss/MoDE_CALVIN_ABC_1` | ≈4.06 | 4.054 |
