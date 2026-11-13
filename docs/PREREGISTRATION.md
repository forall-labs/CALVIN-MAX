# Max-scale pre-registration (frozen for Max-A)

Frozen 2026-11-04 (D-099), before the first Max-A case ran; the freeze commit is recorded in D-099. Written from the Lite exploration (D-034 to D-067) and revised once, as allowed, from the Lite runs of the seven new policies (D-075 to D-098). It replaces the predictions in RESEARCH_PLAN §5 where they differ (Q9's sign flip is withdrawn, D-056). Lite numbers are cited as the basis for each prediction and are not reused as evidence. Any later change is a new dated decision in DECISIONS.md, and results are reported under both versions.

## Design that is fixed

- **Cases: Max-A.** `max_v1` (frozen, D-047), with the fixed preflight (D-053), `core` and `new` tracks. Max-A is a third of Max: in each (track, event, variant, chain position) stratum, the first third of the chains in manifest order (ascending chain id; 20 of 60 for core). `scripts/build_maxA.py` writes `benchmark/manifests/maxA_v1.json` (1,526 cases per policy on 385 chains: Max-A minus its 191 Lite cases) and `maxA_v1_perc.json` (its 887 perceptual core cases). The cut depends only on the frozen manifest. eval_1000's chain ids are a random draw, so it is outcome-blind. Full Max adds the other two thirds of each stratum without rerunning anything. If it is run later, it is reported as an extension, and Max-A stays the confirmatory test.
- **Arms.** Pre and Mid on every Max-A case. `mid_rewind` on the perceptual core cases for every policy (H3). `mid_oracle` on the same cases for the six policies that execute queued actions (FLOWER, MoDE, X-VLA, π0.5, UniVLA, 3D Diffuser Actor), as the secondary queue/state split. GR-1, Seer, RoboFlamingo and OpenVLA-OFT re-plan every step (H = Q = 1), so their oracle arm would equal Mid.
- **Not in Max-A** (D-099): the phase, null and intent tracks, the reversible twins, the magnitude ladders and MoDE seeds 2–3. Their Lite results are reported as exploratory.
- **Policies.** Ten (D-070): FLOWER, MoDE (seed 1), X-VLA, π0.5, OpenVLA-OFT, GR-1, 3D Diffuser Actor, Seer, RoboFlamingo, UniVLA, at the uniform 360-step limit. Admission (D-008, soft since D-087): π0.5, GR-1, Seer, UniVLA and RoboFlamingo reproduce their published scores within ±0.1. 3D Diffuser Actor is budget-limited at 360 steps (1.93; 3.62 at its native 1200, D-090). OpenVLA-OFT has no published score and is admitted at 2.8, below its 3.0 floor (D-098).
- **Held out.** Every Max-A case is outside Lite, so Max-A results are the confirmation. Lite ∪ Max-A ("all cases") is reported beside them, as D-070 requires.
- **Validity.** A case enters a contrast only if it is realized for both arms in that contrast. Preflight failures stay terminal (D-047). Infra errors are rerun, never counted.
- **Statistics.** Event-subtask success, percentage points. 95% chain-cluster bootstrap; policies are resampled jointly on shared chains when pooled. Pooled estimates weight policies equally. Holm correction within each family of per-policy tests. The null-event Δ is each policy's noise floor; effects inside it are not claimed. Every pooled test is also reported without 3D Diffuser Actor (sensitivity, D-099).
- **Analysis code.** `scripts/analyze_maxA.sh` at the freeze commit (`compare_policies.py`, `analyze_decomposition.py`, `analyze_mechanism.py` on `results/cases/<policy>_maxA` and `<policy>_maxA_h3`).

## Confirmatory hypotheses

Power is approximate, from Lite standard errors scaled by √(Max-A held-out cases ÷ Lite cases) ≈ √3.1, one-sided α = 0.05 for directional tests.

| # | Hypothesis | Test | Lite basis | Max-A power |
|---|---|---|---|---|
| H1 | Mid-onset helps perceptual events more than physical ones | Pooled interaction ΔTiming(perceptual) − ΔTiming(physical) > 0 (primary). Secondary: each policy, Holm over ten | Pooled over nine policies +8.9 [+5.3, +12.6]; per policy +2.8 (π0.5) to +14.6 (3DDA), seven of nine with one-sided p < 0.05 (not π0.5, GR-1) | ≈1.00 pooled; ≈0.8 per policy at +9, ≈0.25 at +5 |
| H2 | The physical contrast alone | Pooled physical Mid − Pre, two-sided; report the sign | −2.1 [−5.2, +0.8] over nine policies | ≈0.65 at −2.1 |
| H3 | The perceptual benefit is commitment made before the event | Per policy, Mid − mid_rewind > 0, and mid_rewind − Pre within ±3 points (90% CI inside [−3, +3], i.e. TOST at α = 0.05) | FLOWER, MoDE, X-VLA: Mid − rewind +9.4 / +6.7 / +9.8, residual −3.6 / 0.0 / −1.9. Untested on the seven new policies | ≈1.00 for Mid − rewind where the benefit is ≥ +5; the equivalence needs a residual near 0 |
| H6 | Aftershock | Next-subtask success given the event subtask succeeded in Base and Mid: Mid < Base, per policy | −6.0 to −10.3 (three policies) | ≈0.9 per policy |

H1 is the primary test. H2's sign is reported whatever it is; a result inside ±1 point is reported as "no detectable physical timing effect", not as support. H4 (the benefit tracks degradation) and H5 (perceptual loss happens before contact) need the magnitude ladders and the phase sweep, which Max-A does not run. They move to exploratory, on their Lite evidence.

## Secondary and exploratory (no correction, labelled as such)

- **Where commitment lives.** On the six queue-executing policies, Mid − mid_oracle (queued actions) against mid_oracle − mid_rewind (robot and policy state). Lite: FLOWER state, X-VLA largely queue (camera shift +20.9). Under 2–4× noise it was state on all three (D-067; Lite only).
- **Stale queue under real change, reversible events, target-relocation magnitude, noise onset phase, intent and null tracks, MoDE seeds:** Lite results, descriptive.
- **Chain position and behaviour metrics** on Max-A: descriptive.

## Decision rules

- H1 fails pooled → the paper is a benchmark-and-measurement paper; H3 and H6 are reported as properties of the policies tested.
- H1 holds pooled but fails for a policy → that policy is reported as an exception with its per-event profile; no claim of generality.
- H3 residual not within ±3 points for a policy → the commitment reading does not hold for that policy.
- H1 holds pooled only with 3D Diffuser Actor included → reported as such (D-094: 3DDA is kept for now and may be dropped once all results are in).

## Compute

Max-A ≈ 200 GPU-hours (≈ 400 units at 1.94 units/h) on three L4 runtimes, policies split by runtime: `calvin` RoboFlamingo with 3DDA beside it, then Seer; `calvin2` OpenVLA-OFT (after its Lite), UniVLA, X-VLA; `calvin3` π0.5, GR-1, FLOWER, MoDE.
