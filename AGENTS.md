# ScreenRestore 开发约束

## 架构边界

- `src/screenrestore/core` 只定义文档、参数、算子协议、流水线、缓存和取消机制，不依赖 Qt。
- `src/screenrestore/operators` 中每个恢复步骤都是独立算子；新算子必须有可序列化参数模型和测试。
- `src/screenrestore/ui` 只负责交互与展示，图像算法不得写在 QWidget/QObject 中。
- `src/screenrestore/io` 负责图像、元数据和项目文件；GUI 与 CLI 必须调用同一流水线。
- `src/screenrestore/inference` 仅放可选模型后端；核心模块不得导入可选运行时。
- 原图只读。处理结果、预览代理和导出图像必须是独立对象，任何算子不得原地修改输入。

## 图像约定

- 模块间统一传递 `numpy.ndarray` RGB 图像，形状为 `H×W×3`。
- 读取后的工作格式为 `uint8`；需要色调计算时显式转换为 `[0, 1]` 的 `float32`，输出前裁剪并转换回 `uint8`。
- 仅在调用 OpenCV 的 BGR/LAB/YCrCb/HSV API 边界处显式转换，并在变量名或注释中标明颜色空间。
- Alpha 通道不进入恢复流水线；加载器单独保留元数据，导出器按格式能力处理。

## UI 线程规则

- Qt 主线程只更新界面，不执行全分辨率恢复、导出或外部模型进程。
- 长任务使用 `QThreadPool` worker；worker 通过信号返回不可变结果，禁止直接访问 QWidget。
- 参数预览使用 debounce 和代理图；关闭窗口前取消并等待任务安全退出。

## 测试要求

- 每次修改代码后至少运行受影响测试；每个里程碑结束运行完整测试。
- 测试图像必须由程序合成，不引入来源不明或受版权保护的样例。
- 几何与恢复测试必须验证数值质量，而不只是“不崩溃”。
- CLI、中文路径、取消、缓存失效、项目文件读写必须有回归测试。

## 依赖与隐私

- Python 版本为 3.11，项目依赖由 `pyproject.toml` 管理。
- 禁止加入强制 CUDA、GPU、PyTorch、TensorFlow、网络服务、遥测或云端依赖。
- 默认无模型文件、无独显时必须可启动并运行完整经典流水线。
- 图像内容不得写入日志、上传网络或作为测试夹具提交。

## 第三方与许可证

- 引入或改写第三方代码前，必须核查其许可证，并更新 `THIRD_PARTY_NOTICES.md` 与 `docs/REFERENCE_PROJECTS.md`。
- 无明确开源许可证的仓库默认只能用于理解论文或产品思路。只有权利人明确确认其
  权属并书面授权当前项目使用时，才可复制；必须记录授权范围、原文件和修改内容，
  且不得把该授权误写成面向所有人的通用开源许可证。
- 不提交模型权重、外部二进制或大型数据集。

## 测试数据

测试数据包含四种场景，覆盖不同的退化机制组合。每种场景的退化并非单一因素，
而是透视、反光、摩尔纹、光晕、色彩偏移等问题的交叉叠加。

---

### 场景 1：后台芭蕾 — 艺术品斜拍（ARTWORK）

| 维度 | 说明 |
|------|------|
| **主要退化** | 镜头畸变、斜拍透视、环境照明不均、轻微反光、颜色偏移 |
| **推荐恢复链** | Lens → Perspective → Illumination（弱） → Color → mild Sharpen |
| **容易做错** | 把画作本来的纹理、色调当成噪声或光照问题处理 |
| **关键约束** | CLAHE OFF、Dehalo OFF、Demoire OFF；色彩忠实度优先于锐度；不自动修改构图比例 |

**设计目标**：摄影式复现（photographic/colorimetric reproduction），而非"让图片更好看"。
油画、摄影作品本身的暗部、低对比度、暖色调是创作意图，不得被"修正"。

**色彩策略**：
- 优先支持灰卡/ColorChecker 参照校正（3×3 color correction matrix）
- Gray World / White Patch / 手动灰点作为无参照时的备选
- AUTO aspect ratio 需在 GUI 显示 `Estimated ratio` 和 `confidence`，不悄悄替用户决定

---

### 场景 2：红发女子 — 电影院荧幕拍摄（CINEMA）

| 维度 | 说明 |
|------|------|
| **主要退化** | Keystone、黑位抬升、雾化（veiling glare）、Bloom/Halo、色彩与对比度损失、拍屏噪声 |
| **推荐恢复链** | Geometry → Banding → Dehaze/Black level → Dehalo（受限） → Color-preserving tone → mild Sharpen |
| **容易做错** | 把电影原本的暗调、轮廓光、暖色调"修正常"；Luma 恢复但 Chroma 被压掉 |

**核心问题**：手机拍电影院的退化是乘性+加性的混合：
- 乘性：亮度变化
- 加性：黑位抬升（ambient + projector + camera 的 additive offset）和雾化

**黑位恢复**：CinemaScope 上下黑条是天然参照——理论 RGB=(0,0,0)，实拍若为 (18,17,16)，可估计
加性偏移 $B$，做 $I' = I - B$ 后再受限 tone curve。当前 CINEMA preset 已采用此策略：
仅在检测到可信暗场时有限修正 black level，且关闭自动白平衡、CLAHE、illumination。

**色度保护**：红发恢复失败（头发从偏红棕变黑）的根因是 Luma 恢复时 Chroma 被同步压缩。
应拆分 Y/Cb/Cr 处理，暗部亮度调整时保护 chroma：$C' = C \cdot f(Y)$，即 `shadow_chroma_protection`。

**Halo 判别**：电影原有 Bloom（人物背光轮廓、太阳 glow）与拍屏 Halo 必须区分。
判断标准：高亮核心存在 + 周围径向扩散 + 高频结构无对应变化 → 三者同时满足才判定为 halo。
当前 Cinema 默认关闭 Dehalo，Display 使用 `auto_gate`。

---

### 场景 3：电脑屏幕 — 电子显示器拍摄（DISPLAY / LED）

| 维度 | 说明 |
|------|------|
| **主要退化** | 摩尔纹、子像素格、彩色条纹、PWM/扫描条带、透视、屏幕白场偏差 |
| **推荐恢复链** | Geometry → Banding → Demoire → Denoise → White-field/Exposure → mild Sharpen |
| **容易做错** | 去摩尔纹过头，把头发、建筑纹理、文字一起磨掉 |

**摩尔纹的物理本质**：不是噪声，而是显示器采样格 $f_{display}$ 与相机 CMOS Bayer CFA $f_{camera}$
的空间采样差拍：$f_{moire} = |f_{display} - f_{camera}|$，产生了源图不存在的新频率。

**当前 Demoire 策略**（已实现）：
- YCrCb 分离，luminance 高频分析 + chroma variation
- Sobel structural edge + structure tensor + coherence
- Periodic energy + bilateral filtering + adaptive processing mask
- FFT spectrum + experimental Gaussian notch（默认关闭）
- 核心判别逻辑：规则屏幕栅格局部方向一致 → coherence 高；头发/树叶/建筑纹理多方向 → coherence 低

**可加强方向**：screen lattice estimator——对 rectified image 做 FFT，检测成对离散峰。
若多块 patch 检测到近似相同的 frequency + orientation + harmonic relation → 确认为显示器结构。
这样 `auto_frequency` 才能真正安全开启。

**PWM / rolling shutter banding**：与摩尔纹不同，是行相关的 gain 变化 $I'(x,y) = g(y) \cdot I(x,y)$。
估计方法：排除高梯度结构 → 多列 robust sampling → 低阶 spline/Fourier fitting → 只去除明显周期成分。
当前 BandingOperator 已独立处理，LED/电子海报 preset 强度高于普通 display。

---

### 场景 4：复古街头 — 透明覆盖层反光（GLOSSY_ARTWORK）

| 维度 | 说明 |
|------|------|
| **主要退化** | 强镜面反射、白色高光、反射层、薄膜褶皱、局部折射、透视 |
| **推荐恢复链** | Geometry → Reflection separation → Multiframe/Inpaint → Illumination → Color |
| **容易做错** | 饱和反光区域原始信息已消失，单图无法真实恢复；CLAHE 先执行会放大反光 |

**退化模型**：$I = T + R$（透射 + 反射），或更真实地 $I = \alpha T + \beta R$。

**按反光强度分层处理**：

| 强度 | 特征 | 处理方法 |
|------|------|----------|
| 轻微 | 低饱和度、高亮、大面积平滑 | HSV mask $M = (V > T_V) \land (S < T_S)$，高光压缩 |
| 中等 | 可见灯/窗/人影等独立反射层 | Gradient-domain separation：$\nabla I = \nabla T + \nabla R$，利用反射更平滑、contrast 更低来估计 |
| 强烈 | RGB=(255,255,255) 饱和 | 信息已丢失，只能用 inpainting 猜测。当前限制：reflection mask 面积 ≤ 8% 才允许 OpenCV inpainting |

**UI 标签建议**：恢复结果应明确标注 `Recovered`（真实观测）、`Estimated`（算法推断）、`Generated`（模型生成），
不能全部叫"恢复"。

**多帧融合（最有效方案）**：同一塑料膜照片拍三张不同角度（-5°/0°/+5°），底层作品几乎不变，
反光位置移动。若某像素在 frame 2/3 中 clean，直接用真实像素恢复。当前 `multiframe_fusion.py`
已支持 phase correlation、ECC、Affine/Homography、ORB+RANSAC 对齐，并生成
`recovered_observation_mask` 和 `unresolved_mask`。建议进一步加入
$reflection\_confidence_i(x)$ 权重，融合时自动排除反光帧。

**流水线顺序**：Reflection 应放在 Geometry 之后、CLAHE 之前。Reflection mask 永远在
Geometry 后的原始 radiometric image 上检测，后续 Exposure/CLAHE 不影响此 mask。

---

### 退化机制交叉矩阵

以上四种场景的退化因素可归纳为以下交叉关系：

| 退化因素 | 艺术品斜拍 | 电影院 | 显示器 | 塑料膜反光 |
|----------|:---:|:---:|:---:|:---:|
| 透视/Keystone | ● | ● | ● | ● |
| 镜头畸变 | ● | ○ | ○ | ○ |
| 照明不均 | ● | ○ | ○ | ○ |
| 反光/眩光 | ○ | ○ | ○ | ● |
| 摩尔纹/像素格 | | | ● | |
| PWM/Banding | | | ● | |
| 黑位抬升/雾化 | | ● | | |
| Bloom/Halo | | ● | | |
| 色彩偏移 | ● | ● | ● | ○ |
| 传感器噪声 | ○ | ● | ● | ○ |

> ● 主要退化　○ 次要退化

这些场景可作为回归测试的 benchmark：每次修改算子后，对四类测试图像分别验证
数值质量（PSNR/SSIM/ΔE）和视觉质量，而不只是"不崩溃"。