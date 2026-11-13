"""Human feasibility audit page (D-120): one local HTML file over the rendered audit frames.

    python scripts/build_audit_page.py            # results/audit/index.html (open in a browser)

Each case shows the static and gripper cameras just before and just after the event (render_qual.py audit-render),
the instruction of the event subtask and what the event changed. The auditor answers whether the instructed subtask
can still be completed from the "after" state by a competent robot. No rollout outcome is shown. Answers are kept in
the browser (localStorage) and exported as audit_answers.json for scripts/analyze_audit.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/audit"

EVENT_TEXT = {
    "target_relocation": "The object the instruction refers to ({object}) was moved to a new place.",
    "receptacle_relocation": "The receptacle the instruction refers to ({object}) was moved to a new place.",
    "obstacle_insertion": "A box was placed on the path between the gripper and the target.",
    "arm_push": "The robot arm was pushed to a displaced pose.",
    "grasp_loss": "The block in the gripper ({object}) was released.",
    "camera_shift": "The static camera was shifted slightly.",
    "sensor_noise_onset": "Noise and an occluding patch were added to the camera image.",
    "illumination_switch": "The lighting changed.",
    "visual_theme_switch": "The colours and textures of the scene changed.",
    "distractor_burst": "Small distractor objects appeared on the table.",
}
EVENT_NAME = {k: k.replace("_", " ").capitalize() for k in EVENT_TEXT}


def main():
    sel = json.loads((A / "audit_selection.json").read_text())
    tasks = {t["case_id"]: t.get("task") for t in json.loads((A / "frames/tasks.json").read_text())}
    cases = []
    for s in sel:
        if not (A / f"frames/{s['case_id']}.jpg").exists():
            continue
        obj = s["payload"].get("object", "") if isinstance(s["payload"], dict) else ""
        task = (tasks.get(s["case_id"]) or "").replace("_", " ")
        cases.append({"id": s["case_id"], "event": EVENT_NAME[s["event"]], "change": EVENT_TEXT[s["event"]].format(
            object=obj.replace("_", " ") or "object"), "task": task, "k": s["k"]})
    html = PAGE.replace("__CASES__", json.dumps(cases))
    (A / "index.html").write_text(html)
    print(f"wrote {A / 'index.html'} with {len(cases)} cases")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feasibility Audit</title>
<style>
:root{--bg:#fafaf8;--panel:#fff;--ink:#1b1b19;--ink2:#5c5b57;--line:#e3e2dc;--ok:#1b8a5a;--bad:#c2410c;--unk:#6b6a65;--acc:#2a6fd6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151514;--panel:#1e1e1c;--ink:#ecebe6;--ink2:#a3a29c;--line:#34332f;--ok:#3fb27f;--bad:#f07a45;--unk:#9a9993;--acc:#6aa0f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:2}
h1{font-size:16px;margin:0;font-weight:600}.bar{flex:1;min-width:160px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.count{color:var(--ink2);font-variant-numeric:tabular-nums}button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:6px 12px;cursor:pointer}
main{max-width:980px;margin:0 auto;padding:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.meta{display:flex;gap:12px;flex-wrap:wrap;color:var(--ink2);font-size:13px;margin-bottom:8px}.task{font-size:20px;font-weight:600;margin:4px 0}
.change{margin:4px 0 12px}.imgs{display:grid;grid-template-columns:1fr;gap:6px}.imgs img{width:100%;max-width:640px;image-rendering:auto;border-radius:6px;border:1px solid var(--line)}
.lab{display:grid;grid-template-columns:1fr 1fr;max-width:640px;color:var(--ink2);font-size:13px;text-align:center}
.q{margin:14px 0 8px;font-weight:600}.ans{display:flex;gap:8px;flex-wrap:wrap}.ans button{min-width:150px;padding:10px 14px}
.ans button.sel[data-v=feasible]{background:var(--ok);color:#fff;border-color:var(--ok)}.ans button.sel[data-v=infeasible]{background:var(--bad);color:#fff;border-color:var(--bad)}
.ans button.sel[data-v=unclear]{background:var(--unk);color:#fff;border-color:var(--unk)}textarea{width:100%;margin-top:10px;font:inherit;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.nav{display:flex;justify-content:space-between;margin-top:12px;gap:8px}.help{color:var(--ink2);font-size:13px;margin-top:14px}kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:12px}
</style></head><body>
<header><h1>Feasibility Audit</h1><div class="bar"><i id="prog"></i></div><span class="count" id="count"></span>
<button id="next-open">Next unanswered</button><button id="export">Export answers</button></header>
<main><div class="card">
<div class="meta"><span id="pos"></span><span id="ev"></span><span id="cid"></span></div>
<div class="task" id="task"></div><div class="change" id="change"></div>
<div class="lab"><span>Before the event</span><span>After the event</span></div>
<div class="imgs"><img id="img" alt="Static camera (top) and gripper camera (bottom), before and after the event"></div>
<div class="q">From the state after the event, could a competent robot still complete the instruction?</div>
<div class="ans"><button data-v="feasible">1 &nbsp;Feasible</button><button data-v="infeasible">2 &nbsp;Infeasible</button><button data-v="unclear">3 &nbsp;Unclear / render problem</button></div>
<textarea id="note" rows="2" placeholder="Optional note (why infeasible, what looks wrong)"></textarea>
<div class="nav"><button id="prev">&larr; Previous</button><button id="nxt">Next &rarr;</button></div>
<div class="help">Keys: <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> answer and advance, <kbd>&larr;</kbd> <kbd>&rarr;</kbd> move. Answers are saved in this browser as you go; use <b>Export answers</b> when done and send the file back.
Judge feasibility only: the target or receptacle is still reachable and usable, the path is not fully blocked, nothing needed is out of reach or fallen off the table. Ignore whether the robot would actually succeed.</div>
</div></main>
<script>
const CASES=__CASES__;const KEY="calvinmax_audit_v1";let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function show(){const c=CASES[i];$("pos").textContent=`Case ${i+1} of ${CASES.length}`;$("ev").textContent=c.event;$("cid").textContent=c.id;
$("task").textContent=`Instruction: “${c.task}” (subtask ${c.k} of 5)`;$("change").textContent=c.change;$("img").src=`frames/${c.id}.jpg`;
const a=ans[c.id]||{};document.querySelectorAll(".ans button").forEach(b=>b.classList.toggle("sel",b.dataset.v===a.v));$("note").value=a.note||"";
const n=Object.keys(ans).filter(k=>ans[k].v).length;$("count").textContent=`${n} / ${CASES.length} answered`;$("prog").style.width=(100*n/CASES.length)+"%"}
function answer(v){const c=CASES[i];ans[c.id]={...(ans[c.id]||{}),v,t:new Date().toISOString()};save();if(i<CASES.length-1)i++;show()}
document.querySelectorAll(".ans button").forEach(b=>b.onclick=()=>answer(b.dataset.v));
$("note").oninput=e=>{const c=CASES[i];ans[c.id]={...(ans[c.id]||{}),note:e.target.value};save()};
$("prev").onclick=()=>{if(i>0)i--;show()};$("nxt").onclick=()=>{if(i<CASES.length-1)i++;show()};
$("next-open").onclick=()=>{const j=CASES.findIndex(c=>!(ans[c.id]&&ans[c.id].v));if(j>=0){i=j;show()}};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;if(e.key==="1")answer("feasible");else if(e.key==="2")answer("infeasible");else if(e.key==="3")answer("unclear");else if(e.key==="ArrowRight")$("nxt").click();else if(e.key==="ArrowLeft")$("prev").click()});
$("export").onclick=()=>{const blob=new Blob([JSON.stringify({exported:new Date().toISOString(),n_cases:CASES.length,answers:ans},null,1)],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="audit_answers.json";a.click()};
const j=CASES.findIndex(c=>!(ans[c.id]&&ans[c.id].v));i=j>=0?j:0;show();
</script></body></html>
"""

if __name__ == "__main__":
    main()
