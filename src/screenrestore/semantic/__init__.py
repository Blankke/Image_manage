"""语义分析层：场景分类、目标定位、结构分割、退化分析。

本模块在现有 ImageOperator 流水线之前运行，输出 SceneContext 供
RestorationPlanner 和下游算子消费。不依赖 Qt。
"""

from .analyzer import SemanticAnalyzer
from .context import LocalizationCandidate, SceneContext
from .planner import RestorationPlan, RestorationPlanner
from .scene_classifier import SceneClassifier, classify_scene

__all__ = [
    "SceneContext",
    "LocalizationCandidate",
    "SemanticAnalyzer",
    "SceneClassifier",
    "classify_scene",
    "RestorationPlanner",
    "RestorationPlan",
]
