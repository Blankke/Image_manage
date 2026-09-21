"""Core ML / ONNX 友好的轻量多任务 QuadLocator-S。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _channels(value: int, width_multiplier: float) -> int:
    return max(8, int(round(value * width_multiplier / 8.0)) * 8)


class ConvNormAct(nn.Sequential):
    """只使用常规卷积、BN 与 SiLU，便于后续 Apple 后端转换。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseBlock(nn.Module):
    """MobileNet 风格深度可分离残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.pointwise(self.depthwise(inputs))
        return output + inputs if self.use_residual else output


class PredictionHead(nn.Sequential):
    """共享 1/4 特征上的轻量输出头。"""

    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__(
            DepthwiseBlock(channels, channels, stride=1),
            nn.Conv2d(channels, outputs, kernel_size=1),
        )


class QuadLocatorS(nn.Module):
    """输出 content/outer 四角热图、mask、boundary、两类 presence 与类别。"""

    def __init__(self, width_multiplier: float = 1.0, class_count: int = 4) -> None:
        super().__init__()
        if not 0.35 <= width_multiplier <= 2.0:
            raise ValueError("width_multiplier 必须位于 0.35..2.0")
        channels = [_channels(value, width_multiplier) for value in (16, 24, 40, 80, 128)]
        self.width_multiplier = width_multiplier
        self.class_count = class_count
        self.stem = ConvNormAct(3, channels[0], stride=2)
        self.stage2 = nn.Sequential(
            DepthwiseBlock(channels[0], channels[1], stride=2),
            DepthwiseBlock(channels[1], channels[1], stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthwiseBlock(channels[1], channels[2], stride=2),
            DepthwiseBlock(channels[2], channels[2], stride=1),
        )
        self.stage4 = nn.Sequential(
            DepthwiseBlock(channels[2], channels[3], stride=2),
            DepthwiseBlock(channels[3], channels[3], stride=1),
        )
        self.stage5 = nn.Sequential(
            DepthwiseBlock(channels[3], channels[4], stride=2),
            DepthwiseBlock(channels[4], channels[4], stride=1),
        )
        fpn_channels = _channels(64, width_multiplier)
        self.lateral2 = nn.Conv2d(channels[1], fpn_channels, 1)
        self.lateral3 = nn.Conv2d(channels[2], fpn_channels, 1)
        self.lateral4 = nn.Conv2d(channels[3], fpn_channels, 1)
        self.lateral5 = nn.Conv2d(channels[4], fpn_channels, 1)
        self.fuse4 = DepthwiseBlock(fpn_channels, fpn_channels, stride=1)
        self.fuse3 = DepthwiseBlock(fpn_channels, fpn_channels, stride=1)
        self.fuse2 = DepthwiseBlock(fpn_channels, fpn_channels, stride=1)
        self.content_corner_head = PredictionHead(fpn_channels, 4)
        # 角点残差分支需要空间感受野来区分内画芯、卡纸和外框；旧 1×1 头只能逐像素
        # 重标已有特征。末层零初始化保证架构升级后的 warm-start 与既有 heatmap
        # 逐值一致，再通过公开 median/tail 门决定是否采用训练结果。
        self.content_corner_residual_head = nn.Sequential(
            DepthwiseBlock(fpn_channels, fpn_channels, stride=1),
            DepthwiseBlock(fpn_channels, fpn_channels, stride=1),
            nn.Conv2d(fpn_channels, 4, kernel_size=1),
        )
        nn.init.zeros_(self.content_corner_residual_head[-1].weight)
        nn.init.zeros_(self.content_corner_residual_head[-1].bias)
        self.outer_corner_head = PredictionHead(fpn_channels, 4)
        self.content_mask_head = PredictionHead(fpn_channels, 1)
        self.boundary_head = PredictionHead(fpn_channels, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_evidence_pool = nn.AdaptiveMaxPool2d(1)
        # ``presence_head`` 始终表示 content/target presence；outer 使用独立头，
        # 避免无外框样本的随机 outer 热图被运行时误解为真实层级。
        self.presence_head = nn.Linear(channels[4], 1)
        self.outer_presence_head = nn.Linear(channels[4], 1)
        self.class_head = nn.Linear(channels[4], class_count)
        # 小目标在全局平均池化中会被背景稀释。最大池化残差头专门表达局部证据，
        # 零初始化保证从旧 checkpoint warm-start 时初始输出逐值不变。
        self.presence_local_head = nn.Linear(channels[4], 1)
        self.class_local_head = nn.Linear(channels[4], class_count)
        nn.init.zeros_(self.presence_local_head.weight)
        nn.init.zeros_(self.presence_local_head.bias)
        nn.init.zeros_(self.class_local_head.weight)
        nn.init.zeros_(self.class_local_head.bias)
        # 类别域差（如显示器 bezel/支架）不应通过共享 backbone 改写几何。这个独立
        # 小分支直接读取整图，最后一层零初始化，架构升级初始输出与旧 checkpoint 一致。
        context_channels = [
            _channels(value, width_multiplier) for value in (16, 24, 40, 64, 96, 128)
        ]
        self.class_context_encoder = nn.Sequential(
            ConvNormAct(3, context_channels[0], stride=2),
            DepthwiseBlock(context_channels[0], context_channels[1], stride=2),
            DepthwiseBlock(context_channels[1], context_channels[2], stride=2),
            DepthwiseBlock(context_channels[2], context_channels[3], stride=2),
            # 继续降采样到 1/64，使分类分支看到完整 bezel、支架和外形关系，
            # 避免只凭局部暗边把画框误判成显示器。
            DepthwiseBlock(context_channels[3], context_channels[4], stride=2),
            DepthwiseBlock(context_channels[4], context_channels[5], stride=2),
        )
        self.class_context_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.class_context_max_pool = nn.AdaptiveMaxPool2d(1)
        # 目标域当前缺口集中在 screen vs 非 screen。标量残差只能调整 screen logit，
        # 因而不会重排 artwork/postcard/none 三者，副作用空间显著小于另一套四分类器。
        self.class_context_head = nn.Linear(context_channels[5] * 2, 1)
        nn.init.zeros_(self.class_context_head.weight)
        nn.init.zeros_(self.class_context_head.bias)
        # 海报/墙面画作与 postcard 的域差同样需要整图语境。该标量头只允许修改
        # artwork logit，避免此前四分类局部头微调连带破坏 screen/none 的判断。
        artwork_hidden = _channels(32, width_multiplier)
        self.artwork_context_head = nn.Sequential(
            nn.Linear(context_channels[5] * 2, artwork_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(artwork_hidden, 1),
        )
        nn.init.zeros_(self.artwork_context_head[-1].weight)
        nn.init.zeros_(self.artwork_context_head[-1].bias)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(image)
        feature2 = self.stage2(stem)
        feature3 = self.stage3(feature2)
        feature4 = self.stage4(feature3)
        feature5 = self.stage5(feature4)
        pyramid5 = self.lateral5(feature5)
        pyramid4 = self.fuse4(
            self.lateral4(feature4)
            + F.interpolate(pyramid5, size=feature4.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid3 = self.fuse3(
            self.lateral3(feature3)
            + F.interpolate(pyramid4, size=feature3.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid2 = self.fuse2(
            self.lateral2(feature2)
            + F.interpolate(pyramid3, size=feature2.shape[-2:], mode="bilinear", align_corners=False)
        )
        pooled = self.global_pool(feature5).flatten(1)
        local_evidence = self.local_evidence_pool(feature5).flatten(1)
        presence_base_logits = self.presence_head(pooled)
        class_base_logits = self.class_head(pooled)
        presence_local_residual = self.presence_local_head(local_evidence)
        class_local_residual = self.class_local_head(local_evidence)
        context_feature = self.class_context_encoder(image)
        context_evidence = torch.cat(
            (
                self.class_context_avg_pool(context_feature).flatten(1),
                self.class_context_max_pool(context_feature).flatten(1),
            ),
            dim=1,
        )
        class_pre_context_logits = class_base_logits + class_local_residual
        screen_context_residual = self.class_context_head(context_evidence)
        zero_context_residual = torch.zeros_like(screen_context_residual)
        class_context_residual = torch.cat(
            (
                zero_context_residual,
                zero_context_residual,
                screen_context_residual,
                zero_context_residual,
            ),
            dim=1,
        )
        class_pre_artwork_context_logits = class_pre_context_logits + class_context_residual
        artwork_context_residual = self.artwork_context_head(context_evidence)
        artwork_residual = torch.cat(
            (
                artwork_context_residual,
                zero_context_residual,
                zero_context_residual,
                zero_context_residual,
            ),
            dim=1,
        )
        content_corner_base = self.content_corner_head(pyramid2)
        return {
            "content_corner_heatmaps": content_corner_base
            + self.content_corner_residual_head(pyramid2),
            "outer_corner_heatmaps": self.outer_corner_head(pyramid2),
            "content_mask_logits": self.content_mask_head(pyramid2),
            "boundary_logits": self.boundary_head(pyramid2),
            "presence_logits": presence_base_logits + presence_local_residual,
            "outer_presence_logits": self.outer_presence_head(pooled),
            "class_logits": class_pre_artwork_context_logits + artwork_residual,
            # 训练期的纠错损失需要辨别旧全局头已经答对的样本；导出 wrapper 只暴露
            # 正式 7 个输出，因此这些审计张量不会改变产品推理契约。
            "presence_base_logits": presence_base_logits,
            "class_base_logits": class_base_logits,
            "class_pre_context_logits": class_pre_context_logits,
            "class_pre_artwork_context_logits": class_pre_artwork_context_logits,
            "content_corner_base_heatmaps": content_corner_base,
        }


_ZERO_MIGRATION_PARAMETERS = frozenset(
    {
        "artwork_context_head.0.weight",
        "artwork_context_head.0.bias",
        "artwork_context_head.2.weight",
        "artwork_context_head.2.bias",
    }
)


def load_quadlocator_state_dict(
    model: QuadLocatorS,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """严格加载权重，并将旧 checkpoint 新增的 artwork 标量头迁移为零残差。

    只允许缺少本次新增且零初始化的两个参数。其他缺失、额外参数或 shape 变化
    都视为真正不兼容，避免宽松加载悄悄掩盖模型契约错误。
    """

    current = model.state_dict()
    provided = set(state_dict)
    expected = set(current)
    missing = expected - provided
    unexpected = provided - expected
    disallowed_missing = missing - _ZERO_MIGRATION_PARAMETERS
    shape_mismatches = sorted(
        name
        for name in expected & provided
        if getattr(state_dict[name], "shape", None) != current[name].shape
    )
    if disallowed_missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "QuadLocator checkpoint 不兼容："
            f"missing={sorted(disallowed_missing)}, "
            f"unexpected={sorted(unexpected)}, "
            f"shape_mismatches={shape_mismatches}"
        )
    migrated = dict(state_dict)
    for name in sorted(missing):
        migrated[name] = current[name].detach().clone()
    model.load_state_dict(migrated, strict=True)
    return tuple(sorted(missing))


class QuadLocatorExportWrapper(nn.Module):
    """将字典输出稳定转换为 ONNX/Core ML 可命名元组。"""

    def __init__(self, model: QuadLocatorS) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(image)
        return (
            output["content_corner_heatmaps"],
            output["outer_corner_heatmaps"],
            output["content_mask_logits"],
            output["boundary_logits"],
            output["presence_logits"],
            output["outer_presence_logits"],
            output["class_logits"],
        )
