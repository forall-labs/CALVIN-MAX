"""Build the CALVIN env, task oracle, and annotations without a dataset folder.

One env per process (D-018): calvin_env's scene objects talk to pybullet client 0, and in-process
resets leak solver history. Callers construct the env once at process start and run one rollout.
"""

from __future__ import annotations

from pathlib import Path


def calvin_env_root() -> Path:
    import calvin_env

    return Path(calvin_env.__file__).resolve().parents[1]


def calvin_root() -> Path:
    return calvin_env_root().parent / "calvin"


def make_env(egl: bool = True, cameras: str = "static_and_gripper", scene: str = "calvin_scene_D_eval"):
    import hydra
    from hydra import compose, initialize_config_dir

    root = calvin_env_root()
    with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
        cfg = compose(
            config_name="config_data_collection",
            overrides=[
                "use_vr=false",
                f"cameras={cameras}",
                f"scene={scene}",
                "robot=panda_longer_finger",
                f"env.use_egl={str(egl).lower()}",
                f"data_path={root / 'data'}",
            ],
        )
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
    for cam in env.cameras:  # nominal intrinsics, for policies that deproject with a fixed calibration (D-079)
        cam.fov0 = cam.fov
    return env


def make_task_oracle():
    import hydra
    from omegaconf import OmegaConf

    return hydra.utils.instantiate(OmegaConf.load(calvin_env_root() / "conf/tasks/new_playtable_tasks.yaml"))


def load_annotations(split: str = "validation"):
    from omegaconf import OmegaConf

    name = "new_playtable_validation.yaml" if split == "validation" else "new_playtable.yaml"
    return OmegaConf.load(calvin_root() / "calvin_models/conf/annotations" / name)


def block_entities(env):
    from calvin_max.rollout import Entity

    return [Entity(o.name, o.uid) for o in env.scene.movable_objects]


def tracked_entities(env):
    """Blocks plus the table links that tasks act on (handles, button, switch, slider shelf)."""
    from calvin_max.rollout import Entity
    from calvin_max.tasks import TABLE_LINKS

    table = env.scene.fixed_objects[0]
    links = {}
    for j in range(env.p.getNumJoints(table.uid, physicsClientId=env.cid)):
        name = env.p.getJointInfo(table.uid, j, physicsClientId=env.cid)[12].decode()
        if name in TABLE_LINKS:
            links[name] = j
    return block_entities(env) + [Entity(n, table.uid, links[n]) for n in TABLE_LINKS]
