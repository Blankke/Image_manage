"""镜头畸变校正与多视角棋盘格标定。

模块仅依赖 OpenCV/NumPy，并保持 RGB ``uint8`` 输入输出。标定结果中的焦距和
主点使用相对图像尺寸的归一化值，因此可以安全地用于同一镜头拍摄的不同分辨率。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_rgb_float


class LensModel(StrEnum):
    """OpenCV 支持的两种镜头模型。"""

    PINHOLE = "pinhole"
    FISHEYE = "fisheye"


@dataclass
class LensDistortionParameters(ParameterModel):
    """可跨分辨率复用的镜头参数。

    ``focal_x/focal_y`` 分别相对于图像宽高归一化，``principal_x/principal_y``
    位于 ``[0, 1]``。针孔模型使用 ``k1/k2/p1/p2/k3``，鱼眼模型使用
    ``k1/k2/k3/k4``。
    """

    model: LensModel = LensModel.PINHOLE
    focal_x: float = 1.0
    focal_y: float = 1.0
    principal_x: float = 0.5
    principal_y: float = 0.5
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    optimize_camera_matrix: bool = False
    crop_balance: float = 0.0
    crop_to_valid: bool = False

    def validate(self) -> None:
        scalars = (
            self.focal_x,
            self.focal_y,
            self.principal_x,
            self.principal_y,
            self.k1,
            self.k2,
            self.p1,
            self.p2,
            self.k3,
            self.k4,
            self.crop_balance,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("镜头参数必须是有限数值")
        if not 0.05 <= self.focal_x <= 20.0 or not 0.05 <= self.focal_y <= 20.0:
            raise ValueError("归一化焦距必须位于 0.05..20")
        if not 0.0 <= self.principal_x <= 1.0 or not 0.0 <= self.principal_y <= 1.0:
            raise ValueError("镜头主点必须位于 [0, 1]")
        if max(abs(value) for value in (self.k1, self.k2, self.k3, self.k4)) > 5.0:
            raise ValueError("径向畸变系数绝对值不能超过 5")
        if max(abs(self.p1), abs(self.p2)) > 1.0:
            raise ValueError("切向畸变系数绝对值不能超过 1")
        if not 0.0 <= self.crop_balance <= 1.0:
            raise ValueError("镜头裁切平衡必须位于 [0, 1]")


@dataclass
class LensCalibrationParameters(ParameterModel):
    """棋盘格内角点标定设置。"""

    board_columns: int = 9
    board_rows: int = 6
    square_size: float = 1.0
    min_views: int = 5
    model: LensModel = LensModel.PINHOLE

    def validate(self) -> None:
        if not 3 <= self.board_columns <= 30 or not 3 <= self.board_rows <= 30:
            raise ValueError("棋盘格横纵内角点数量必须位于 3..30")
        if not np.isfinite(self.square_size) or self.square_size <= 0:
            raise ValueError("棋盘格方格尺寸必须大于 0")
        if not 3 <= self.min_views <= 50:
            raise ValueError("镜头标定至少需要 3 张、最多使用 50 张图")


@dataclass(frozen=True, slots=True)
class LensCalibrationResult:
    """镜头标定结果及可解释误差。"""

    parameters: LensDistortionParameters
    rms_error: float
    per_view_errors: tuple[float, ...]
    used_views: int
    rejected_views: int
    image_size: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        """转换为 Web/项目文件可直接序列化的结构。"""

        return {
            "parameters": self.parameters.to_dict(),
            "rms_error": self.rms_error,
            "per_view_errors": list(self.per_view_errors),
            "used_views": self.used_views,
            "rejected_views": self.rejected_views,
            "image_size": list(self.image_size),
        }


def camera_matrix_for_shape(
    params: LensDistortionParameters,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """从归一化参数构造当前分辨率的 3×3 内参矩阵。"""

    height, width = int(image_shape[0]), int(image_shape[1])
    return np.array(
        [
            [params.focal_x * width, 0.0, params.principal_x * (width - 1)],
            [0.0, params.focal_y * height, params.principal_y * (height - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def undistort_lens(
    image_rgb: np.ndarray,
    params: LensDistortionParameters,
) -> tuple[np.ndarray, dict[str, object]]:
    """执行针孔或鱼眼镜头校正，返回独立 RGB 图像和映射元数据。"""

    _validate_rgb(image_rgb)
    params.validate()
    height, width = image_rgb.shape[:2]
    size = (width, height)
    camera = camera_matrix_for_shape(params, image_rgb.shape)
    identity = np.eye(3, dtype=np.float64)

    if params.model == LensModel.FISHEYE:
        distortion = np.array([params.k1, params.k2, params.k3, params.k4], np.float64)
        new_camera = (
            cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera,
                distortion,
                size,
                identity,
                balance=params.crop_balance,
                new_size=size,
            )
            if params.optimize_camera_matrix
            else camera.copy()
        )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            camera,
            distortion,
            identity,
            new_camera,
            size,
            cv2.CV_32FC1,
        )
        roi = (0, 0, width, height)
    else:
        distortion = np.array(
            [params.k1, params.k2, params.p1, params.p2, params.k3],
            dtype=np.float64,
        )
        if params.optimize_camera_matrix:
            new_camera, roi = cv2.getOptimalNewCameraMatrix(
                camera,
                distortion,
                size,
                params.crop_balance,
                size,
            )
        else:
            new_camera = camera.copy()
            roi = (0, 0, width, height)
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera,
            distortion,
            None,
            new_camera,
            size,
            cv2.CV_32FC1,
        )

    corrected = cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    crop_x, crop_y, crop_width, crop_height = (int(value) for value in roi)
    if params.crop_to_valid and crop_width > 1 and crop_height > 1:
        corrected = corrected[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width].copy()
    metadata: dict[str, object] = {
        "model": params.model.value,
        "camera_matrix": camera.tolist(),
        "new_camera_matrix": new_camera.tolist(),
        "valid_roi": [crop_x, crop_y, crop_width, crop_height],
        "cropped": params.crop_to_valid,
    }
    return np.ascontiguousarray(corrected), metadata


def calibrate_lens(
    images_rgb: list[np.ndarray],
    params: LensCalibrationParameters | None = None,
) -> LensCalibrationResult:
    """从多张不同姿态的棋盘格照片估计镜头内参与畸变。

    无法检测完整内角点的图片会被拒绝；有效视角不足时明确失败，不会输出看似
    可用但不稳定的参数。
    """

    settings = params or LensCalibrationParameters()
    settings.validate()
    if len(images_rgb) < settings.min_views:
        raise ValueError(f"镜头标定至少需要 {settings.min_views} 张棋盘格照片")
    if len(images_rgb) > 50:
        raise ValueError("一次镜头标定最多接受 50 张图片")

    first_height, first_width = images_rgb[0].shape[:2]
    image_size = (first_width, first_height)
    pattern_size = (settings.board_columns, settings.board_rows)
    object_template = np.zeros((settings.board_rows * settings.board_columns, 3), np.float32)
    object_template[:, :2] = (
        np.mgrid[0 : settings.board_columns, 0 : settings.board_rows]
        .T.reshape(-1, 2)
        .astype(np.float32)
        * settings.square_size
    )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    for image in images_rgb:
        _validate_rgb(image)
        if image.shape[:2] != (first_height, first_width):
            raise ValueError("同一次镜头标定的图片尺寸必须一致")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found or corners is None:
            continue
        object_points.append(object_template.copy())
        image_points.append(corners.astype(np.float32))

    if len(image_points) < settings.min_views:
        raise ValueError(
            f"只在 {len(image_points)} 张图片中检测到完整棋盘格，至少需要 {settings.min_views} 张"
        )

    if settings.model == LensModel.FISHEYE:
        rms, camera, distortion, rotations, translations = _calibrate_fisheye(
            object_points,
            image_points,
            image_size,
        )
    else:
        rms, camera, distortion, rotations, translations = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    per_view = _calibration_errors(
        object_points,
        image_points,
        rotations,
        translations,
        camera,
        distortion,
        settings.model,
    )
    flat_distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if settings.model == LensModel.FISHEYE:
        lens_params = LensDistortionParameters(
            model=settings.model,
            focal_x=float(camera[0, 0] / first_width),
            focal_y=float(camera[1, 1] / first_height),
            principal_x=float(camera[0, 2] / max(1, first_width - 1)),
            principal_y=float(camera[1, 2] / max(1, first_height - 1)),
            k1=float(flat_distortion[0]),
            k2=float(flat_distortion[1]),
            k3=float(flat_distortion[2]),
            k4=float(flat_distortion[3]),
        )
    else:
        padded = np.pad(flat_distortion, (0, max(0, 5 - len(flat_distortion))))
        lens_params = LensDistortionParameters(
            model=settings.model,
            focal_x=float(camera[0, 0] / first_width),
            focal_y=float(camera[1, 1] / first_height),
            principal_x=float(camera[0, 2] / max(1, first_width - 1)),
            principal_y=float(camera[1, 2] / max(1, first_height - 1)),
            k1=float(padded[0]),
            k2=float(padded[1]),
            p1=float(padded[2]),
            p2=float(padded[3]),
            k3=float(padded[4]),
        )
    lens_params.validate()
    return LensCalibrationResult(
        parameters=lens_params,
        rms_error=float(rms),
        per_view_errors=tuple(float(value) for value in per_view),
        used_views=len(image_points),
        rejected_views=len(images_rgb) - len(image_points),
        image_size=image_size,
    )


class LensDistortionOperator(ImageOperator[LensDistortionParameters]):
    """在透视恢复前执行可选镜头畸变校正。"""

    id = "lens_distortion"
    display_name = "镜头畸变校正"
    parameter_type = LensDistortionParameters
    reorderable = False

    def default_parameters(self) -> LensDistortionParameters:
        return LensDistortionParameters()

    def apply(
        self,
        image: np.ndarray,
        params: LensDistortionParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        context.report(0.05, "准备镜头畸变校正")
        corrected, metadata = undistort_lens(image, params)
        context.metadata["lens_distortion"] = metadata
        context.report(1.0, "镜头畸变校正完成")
        return clip_float(corrected)


def _calibrate_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """把普通点数组调整为 OpenCV fisheye API 要求的形状。"""

    fisheye_objects = [points.reshape(1, -1, 3).astype(np.float64) for points in object_points]
    fisheye_images = [points.reshape(1, -1, 2).astype(np.float64) for points in image_points]
    camera = np.array(
        [
            [max(image_size), 0.0, (image_size[0] - 1) / 2],
            [0.0, max(image_size), (image_size[1] - 1) / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), np.float64)
    rotations = [np.zeros((1, 1, 3), np.float64) for _ in fisheye_objects]
    translations = [np.zeros((1, 1, 3), np.float64) for _ in fisheye_objects]
    rms, camera, distortion, rotations, translations = cv2.fisheye.calibrate(
        fisheye_objects,
        fisheye_images,
        image_size,
        camera,
        distortion,
        rotations,
        translations,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
    )
    return float(rms), camera, distortion, tuple(rotations), tuple(translations)


def _calibration_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotations: tuple[np.ndarray, ...] | list[np.ndarray],
    translations: tuple[np.ndarray, ...] | list[np.ndarray],
    camera: np.ndarray,
    distortion: np.ndarray,
    model: LensModel,
) -> list[float]:
    """计算每个有效视角的像素 RMS 重投影误差。"""

    errors: list[float] = []
    for objects, observed, rotation, translation in zip(
        object_points,
        image_points,
        rotations,
        translations,
        strict=True,
    ):
        if model == LensModel.FISHEYE:
            projected, _ = cv2.fisheye.projectPoints(
                objects.reshape(1, -1, 3).astype(np.float64),
                np.asarray(rotation).reshape(3, 1),
                np.asarray(translation).reshape(3, 1),
                camera,
                distortion,
            )
        else:
            projected, _ = cv2.projectPoints(
                objects,
                rotation,
                translation,
                camera,
                distortion,
            )
        difference = projected.reshape(-1, 2) - observed.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(np.square(difference), axis=1)))))
    return errors


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.float32):
        raise ValueError("镜头模块需要 H×W×3 RGB uint8/float32 图像")
