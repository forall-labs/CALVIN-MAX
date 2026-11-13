"""Diff our composed env config against an official dataset's validation/.hydra/merged_config.yaml.

    python compare_env_config.py /content/data/calvin_debug_dataset/validation/.hydra/merged_config.yaml

Policies are evaluated in the env built from the dataset config (CALVIN's get_env), so any
difference in cameras, robot, scene or env parameters must be resolved before Base numbers count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from calvin_max.envs import calvin_env_root


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        out[prefix[:-1]] = json.dumps(d)
    else:
        out[prefix[:-1]] = d
    return out


def main():
    official = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=False)
    root = calvin_env_root()
    with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
        ours = compose(
            config_name="config_data_collection",
            overrides=["use_vr=false", "cameras=static_and_gripper", "scene=calvin_scene_D_eval", "robot=panda_longer_finger"],
        )
    ours = OmegaConf.to_container(ours, resolve=False)
    for section in ("cameras", "robot", "scene", "env", "tasks"):
        a, b = flat(official.get(section, {})), flat(ours.get(section, {}))
        diffs = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        print(f"== {section}: {len(diffs)} differences")
        for k, (x, y) in diffs.items():
            print(f"  {k}: official={x!r} ours={y!r}")
    print("official top-level keys:", sorted(official))


if __name__ == "__main__":
    main()
