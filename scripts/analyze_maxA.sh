#!/usr/bin/env bash
# Confirmatory Max-A analysis (PREREG_MAX, D-099), frozen with the pre-registration. Run locally on the mirror.
#   bash scripts/analyze_maxA.sh
# Writes results/analysis/maxA_*.json:
#   q9_heldout     H1, H2: Max-A cases only (all held out from Lite); the confirmation
#   q9_heldout_no3dda  the same without 3D Diffuser Actor (sensitivity, D-099)
#   q9_all         Lite + Max-A (reported beside the confirmation, D-070)
#   decomposition  H3: Pre/Mid from <p>_maxA, mid_rewind and mid_oracle from <p>_maxA_h3
#   mechanism      H6 aftershock (and the queue-length breakdown), Max-A cases
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
R=results/cases
A=results/analysis
NAMES=(FLOWER MoDE X-VLA pi0.5 GR-1 Seer UniVLA 3DDA RoboFlamingo OpenVLA-OFT)
DIRS=(flower mode xvla pi05 gr1 seer univla dda rf oft)
LITE_flower="$R/flower_lite_v0 $R/flower_lite_v1 $R/flower_lite_v1b $R/flower_lite_v1c $R/flower_lite_d048"
LITE_mode="$R/mode1_lite_v1 $R/mode1_lite_v1c $R/mode1_lite_d048"
LITE_xvla="$R/xvla_lite_v1 $R/xvla_lite_d048"
held=(); held_no=(); all=(); dec=(); mech=()
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}; d=${DIRS[$i]}
  m=$R/${d}_maxA/results.jsonl; h=$R/${d}_maxA_h3/results.jsonl
  held+=(--policy "$n" "$m"); [ "$d" = dda ] || held_no+=(--policy "$n" "$m")
  lv=LITE_$d; lite=${!lv:-$R/${d}_lite_v1}
  all+=(--policy "$n" $(for x in $lite; do echo "$x/results.jsonl"; done) "$m")
  dec+=(--policy "$n" --native "$m" --variants "$h")
  mech+=(--policy "$n" "$m")
done
python3 scripts/compare_policies.py "${held[@]}" --out $A/maxA_q9_heldout.json
python3 scripts/compare_policies.py "${held_no[@]}" --out $A/maxA_q9_heldout_no3dda.json
python3 scripts/compare_policies.py "${all[@]}" --out $A/maxA_q9_all.json
python3 scripts/analyze_decomposition.py "${dec[@]}" --out $A/maxA_decomposition.json
python3 scripts/analyze_mechanism.py "${mech[@]}" --out $A/maxA_mechanism.json
