# CALVIN facts from the code

Read from source on 2026-10-28. Pins:
- `calvin_env` @ `797142c588c21e76717268b7b430958dbd13bf48` (2022-12-23)
- `calvin` @ `fa03f01f19c65920e18cf37398a9ce859274af76` (2025-09-08)

Items marked **[verify]** still need a run.

## Evaluator (`calvin_models/calvin_agent/evaluation/evaluate_policy.py`)
- `EP_LEN = 360` steps per subtask; `NUM_SEQUENCES = 1000`; `seed_everything(0)`.
- `get_sequences()` samples 1,000 chains over 8 symbolic start conditions (led, lightbulb, slider, drawer, block locations, grasped), under `temp_seed(0)`. The chains come from multiprocessing, so we freeze the list once into our manifest rather than regenerating it.
- `make_env(dataset_path)` builds the env from `dataset_path/validation/.hydra/merged_config.yaml`, so we need an env-D validation folder, not the 517 GB training set. **[verify]** whether the debug dataset's config equals the D eval config.
- Per subtask:
  1. `model.reset()`
  2. `start_info = env.get_info()`
  3. step up to 360 times; each step checks `task_oracle.get_task_info_for_set(start_info, current_info, {subtask})`

## Instructions
- Eval uses exactly one fixed string per task, from `conf/annotations/new_playtable_validation.yaml`.
- Training annotations (`new_playtable.yaml`) have 389 strings: 7–20 per task, all 34 tasks.

`paraphrase_swap` variant A therefore takes a training paraphrase (seen). Variant B is a frozen, hand-written novel paraphrase.

## Oracle (`calvin_env/envs/tasks.py`, `conf/tasks/new_playtable_tasks.yaml`)
- `get_task_info(start, end)` returns **every** task satisfied between two infos. This is what powers our side-effect and undo metrics.
- Thresholds that matter for event magnitudes:

  | Task | Threshold |
  |---|---|
  | push | ±0.10 m in x |
  | rotate | ±60° |
  | lift | +0.05 m (table, drawer) / +0.03 m (slider) |
  | slider | ±0.15 of its 0–0.56 range |
  | drawer | ±0.12 of its 0–0.24 range |

  A 6–12 cm block teleport can therefore fire a false `push_*` success. The `start_info` rebase is mandatory.

## State
- `robot_obs` (15 values with Euler orientation; quaternion layout is 16):
  - tcp_pos[0:3]
  - tcp_orn[3:6]
  - gripper_width[6]
  - arm joints[7:14]
  - gripper_action[14]
- `scene_obs` order:
  - doors (slide, drawer)
  - button joint
  - switch joint
  - lights (lightbulb, led)
  - 3 block poses (pos + orientation)

  Confirmed from `get_env_state_for_initial_condition`: 24 values, Euler.
  - [0] slider, [1] drawer, [2] button, [3] switch, [4] lightbulb, [5] LED
  - [6:12] red, [12:18] blue, [18:24] pink (pos3 + euler3)
- Hidden state that is **not** in obs:
  - joint and object velocities
  - `Button.prev_is_pressed` / logical state
  - `Robot.target_pos/orn`

  Restoring from `(robot_obs, scene_obs)` is therefore *not* an exact state restore. Pre and Mid arms are reached by **replaying actions from the initial reset** in a fresh process. In-process `saveState`/`restoreState` is *not* exact (D-018) and is not used.
- Determinism (measured on Colab, D-018): a fresh process reproduces runs bit-for-bit, rendered or not. In-process resets do not: up to ~1 cm drift, or 0.19 mm even after restoreState plus all Python-side state. calvin_env supports only one env per process. Light on/off state is Python-side too.
- The argument-free `env.reset()` runs a hidden `stepSimulation` (`reset_movable_objects`), and CALVIN's official evaluator reuses one env across chains. Official CALVIN numbers therefore depend slightly on chain order.
- `scene.reset_movable_objects()` calls `p.stepSimulation()` without `physicsClientId`. That only matters for random resets, which the evaluator does not use.
- `env.step(action)` runs `robot.apply_action(action)`, then `action_repeat` × `stepSimulation`, then `scene.step()` (button/switch logic).

## Mechanics we exploit
- The button is a **toggle**: each new press flips the LED (`button.py`). The switch drives the lightbulb.

  `goal_preemption` on `turn_on_led` therefore makes blind continuation *harmful*: the next press turns the LED back off.
- Cameras configs available: static+gripper, static+tactile, static+gripper+tactile (DIGIT via `tacto`). Depth comes from the same renders.
- Scene D: table surface box `[[0.0,-0.15,0.46],[0.35,-0.03,0.46]]`; slider sub-surfaces left/right; blocks are red (middle size), blue (small), pink (big).
