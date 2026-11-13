# Vendored from RLinf (https://github.com/RLinf/RLinf @ d80cf00), Apache License 2.0, Copyright 2025 The RLinf
# Authors: rlinf/models/embodiment/openpi/policies/calvin_policy.py (CalvinInputs/CalvinOutputs),
# rlinf/models/embodiment/openpi/dataconfig/calvin_dataconfig.py (LeRobotCalvinDataConfig) and the `pi05_calvin`
# TrainConfig from rlinf/models/embodiment/openpi/dataconfig/__init__.py. Copied so the pi0.5 server needs only
# the `rlinf-openpi` package, not RLinf's runtime (D-071). Logic unchanged; unused training-only parts dropped.
"""RLinf's openpi CALVIN transforms and the pi05_calvin config, for scripts/adapters/pi05_server.py."""

import dataclasses
import pathlib

import einops
import numpy as np
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.config import AssetsConfig, DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CalvinInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state = np.concatenate([data["observation/state_ee_pos"], data["observation/state_ee_rot"],
                                data["observation/state_gripper"]])
        inputs = {
            "state": state,
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": wrist_image,
                      "right_wrist_0_rgb": np.zeros_like(base_image)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_},
        }
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LeRobotCalvinDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(inputs=[CalvinInputs(model_type=model_config.model_type)],
                                            outputs=[CalvinOutputs()])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.delta_ee_pos", "action.delta_ee_rot", "action.gripper"),
        )


PI05_CALVIN = TrainConfig(
    name="pi05_calvin",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=5, discrete_state_input=False),
    data=LeRobotCalvinDataConfig(
        repo_id="InternRobotics/InternData-Calvin_ABC",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_calvin/assets"),
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi05_base/params"),
    pytorch_weight_path="checkpoints/torch/pi05_base",
    num_train_steps=30_000,
)
