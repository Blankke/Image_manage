"""P3 强专项恢复路由；一次最多自动启用一个 specialist。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class RestorationRoute(StrEnum):
    CLEAN_BYPASS = "clean_bypass"
    DEMOIRE = "demoire"
    REFLECTION = "reflection"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ArtifactRoutingDecision:
    route: RestorationRoute
    labels: tuple[str, ...]
    severity: dict[str, float]
    reason: str


def route_artifacts(
    probabilities: Mapping[str, float],
    severity: Mapping[str, float],
    *,
    probability_threshold: float = 0.65,
    strong_threshold: float = 0.55,
) -> ArtifactRoutingDecision:
    """clean bypass；单一强 artifact 进入专项；混合强 artifact 进入 review。"""

    artifact_labels = ("noise", "blur", "jpeg", "photometric", "reflection", "moire", "dewarp")
    severity_values = {name: float(severity.get(name, 0.0)) for name in artifact_labels}
    detected = tuple(
        name for name in artifact_labels if float(probabilities.get(name, 0.0)) >= probability_threshold
    )
    selected_specialists = tuple(
        name
        for name in ("moire", "reflection")
        if float(probabilities.get(name, 0.0)) >= probability_threshold
        and float(severity.get(name, 0.0)) >= strong_threshold
    )
    dewarp_strong = (
        float(probabilities.get("dewarp", 0.0)) >= probability_threshold
        and float(severity.get("dewarp", 0.0)) >= strong_threshold
    )
    if len(selected_specialists) > 1 or dewarp_strong:
        return ArtifactRoutingDecision(
            RestorationRoute.REVIEW,
            detected,
            severity_values,
            "复杂几何或多个强 artifact 需要保守复核",
        )
    if selected_specialists == ("moire",):
        return ArtifactRoutingDecision(RestorationRoute.DEMOIRE, detected, severity_values, "检测到单一强摩尔纹")
    if selected_specialists == ("reflection",):
        return ArtifactRoutingDecision(RestorationRoute.REFLECTION, detected, severity_values, "检测到单一强反射")
    return ArtifactRoutingDecision(RestorationRoute.CLEAN_BYPASS, detected, severity_values, "未发现需要强专项处理的证据")


__all__ = ["ArtifactRoutingDecision", "RestorationRoute", "route_artifacts"]
