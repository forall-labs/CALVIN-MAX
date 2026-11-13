#!/usr/bin/env bash
# Interim Max-A analysis (D-116): the frozen analyze_maxA.sh restricted to the policies whose runs exist, written to
# results/analysis/maxA_interim_*.json so the confirmatory files are produced only once, by the frozen script, on
# all ten. Same scripts, same arguments, same statistics; a policy is dropped from a family when its folder is
# missing. Partial folders (runs still going) enter as they stand and are listed in the output's header line.
#   bash scripts/analyze_maxA_interim.sh "flower mode xvla ..." ["flower mode ..."]   # Mid/Pre set, H3 set
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
R=results/cases
A=results/analysis
declare -a NAMES=(FLOWER MoDE X-VLA pi0.5 GR-1 Seer UniVLA 3DDA RoboFlamingo OpenVLA-OFT)
declare -a DIRS=(flower mode xvla pi05 gr1 seer univla dda rf oft)
MAIN=" ${1:-flower mode xvla pi05 gr1 dda rf oft} "; H3=" ${2:-flower mode xvla gr1 dda rf} "
LITE_flower="$R/flower_lite_v0 $R/flower_lite_v1 $R/flower_lite_v1b $R/flower_lite_v1c $R/flower_lite_d048"
LITE_mode="$R/mode1_lite_v1 $R/mode1_lite_v1c $R/mode1_lite_d048"
LITE_xvla="$R/xvla_lite_v1 $R/xvla_lite_d048"
held=(); held_no=(); all=(); dec=(); mech=()
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}; d=${DIRS[$i]}
  m=$R/${d}_maxA/results.jsonl; h=$R/${d}_maxA_h3/results.jsonl
  if [[ $MAIN == *" $d "* ]]; then
    held+=(--policy "$n" "$m"); [ "$d" = dda ] || held_no+=(--policy "$n" "$m")
    lv=LITE_$d; lite=${!lv:-$R/${d}_lite_v1}
    all+=(--policy "$n" $(for x in $lite; do echo "$x/results.jsonl"; done) "$m")
    mech+=(--policy "$n" "$m")
  fi
  [[ $H3 == *" $d "* ]] && dec+=(--policy "$n" --native "$m" --variants "$h")
done
echo "interim: Mid/Pre [$MAIN] H3 [$H3]"
python3 scripts/compare_policies.py "${held[@]}" --out $A/maxA_interim_q9_heldout.json
python3 scripts/compare_policies.py "${held_no[@]}" --out $A/maxA_interim_q9_heldout_no3dda.json
python3 scripts/compare_policies.py "${all[@]}" --out $A/maxA_interim_q9_all.json
python3 scripts/analyze_decomposition.py "${dec[@]}" --out $A/maxA_interim_decomposition.json
python3 scripts/analyze_mechanism.py "${mech[@]}" --out $A/maxA_interim_mechanism.json
