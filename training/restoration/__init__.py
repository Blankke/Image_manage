"""Fidelity 恢复训练：DIV2K HR 在线相机退化、有界残差模型与验证工具。"""

from .degradation import CameraDegradationConfig, CameraDegradationSample, degrade_camera_image
from .model import BoundedResidualNet

__all__ = [
    "BoundedResidualNet",
    "CameraDegradationConfig",
    "CameraDegradationSample",
    "degrade_camera_image",
]
