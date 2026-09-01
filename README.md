# ScreenRestore

ScreenRestore 是一款完全离线的平面图像电子化恢复工具，用于处理手机斜拍画作、明信片、显示器、电子海报、投影幕布、电影院银幕及 LED 大屏。它以可拒绝的内容层定位和平面透视恢复为起点，继续处理光度偏差、条带、摩尔纹、噪声和反光，并区分观测恢复与生成增强。

应用不会上传图片，不包含遥测，不调用云端 API。没有独立显卡、模型文件、PyTorch 或 CUDA 时，经典 CPU 流水线仍可完整启动和运行。

## 当前能力

- PNG、JPEG、WebP、BMP、TIFF；EXIF Orientation；中文路径；最长边 1600 像素代理预览。
- 同源本地 Web 前后端：多图拖放、四角与 Mesh 拖动、镜头标定、Geometry / Fidelity / AI Enhanced 三路输出、拖动分割比较、结果下载和 JSON 诊断；默认只监听 `127.0.0.1`，上传图片只驻留请求内存。
- CLI、Web、GUI 共用 `AutomaticGeometryService`：可选 QuadLocator ONNX 负责 content/outer 语义，原分辨率直线精修负责像素落点，统一置信度策略负责接受或拒绝。无训练模型时的经典候选仅作保守 fallback 和诊断。
- 针孔/鱼眼镜头畸变校正；5 张以上不同姿态棋盘格的多视角标定；归一化内参可跨同镜头分辨率复用。
- 2～15 行列可编辑控制网格、二维样条逆映射和弯曲银幕校正；透视与非线性形变分别建模。
- 2～20 张多帧对齐，支持平移/仿射/单应模型；按裁切、反光、时域离群和局部清晰度融合真实观测，并单独报告“其他帧补回”和“所有帧仍未解决”的比例。
- 带弱相机标定的 AUTO 物理比例恢复；自动/估计/自由、16:9、16:10、4:3、1.85:1、2.39:1 和自定义比例；纵向纸张不会被强制横向拉伸。
- 非破坏式可序列化流水线；算子启停与安全重排；节点 LRU 缓存；撤销/重做；异步 debounce 预览和取消。
- 周期条带与影院宽光幕校正、方向相干的亮度/色度联合去摩尔纹、结构边缘保护、实验性 Gaussian notch、摩尔纹处理热图、证据门控的高光扩散光晕抑制、反光蒙版及小区域修复。
- 影院 Fidelity 默认禁用白平衡、CLAHE、照明和实验性 DCT；只在检测到可信暗场时启用有限幅自适应黑位与轻微对比恢复，避免把电影固有色温、反射构图或暗部当成缺陷。
- DISPLAY Fidelity 只在检测到占画面至少 65% 的低饱和亮背景时启用线性光白场校正；普通照片保持中性，白底屏摄不再停留在整体灰雾状态。
- Gray World、White Patch、手动 RGB 增益、中性灰点选取；完整曝光/色调；LAB CLAHE；光照场校正；四种降噪；Unsharp Mask；实验性 Wiener 反卷积。
- 显示器、电子海报、电影院/投影、LED 大屏、文档/PPT、自定义预设。
- 原图/结果左右并排和可拖分割条；按住空格临时看原图；Fit/100%/200%；RGB/HSV；直方图、热图和频谱。
- `.screenrestore.json` 项目；相对源图路径、SHA-256 核对和重新定位；全分辨率后台 PNG/JPEG/WebP/TIFF 导出。
- 与 GUI 共用流水线的 CLI；JSON 诊断；滚动日志；PyInstaller Windows 配置。
- 语义独立的 Restoration 与 Enhancement 模型节点、外部程序、ONNX Runtime、OpenVINO 后端及无接缝 float32 tile 融合；提供 NAFNet/Real-ESRGAN 移植脚本和可复现安装器。核心包不附带权重、PyTorch 或外部二进制。

## 本地 Web 工具

Web 版不需要 Node.js、npm、云服务或额外 Web 框架，安装项目依赖后直接运行：

```bash
source .venv/bin/activate
which python
python -m screenrestore.web.server --open
```

也可以在重新安装当前项目后使用入口命令：

```bash
screenrestore-web --host 127.0.0.1 --port 8765 --open
```

浏览器访问 `http://127.0.0.1:8765/`。工作流程是：

1. 拖入一张照片，或同一静态内容的 2～8 张连拍。
2. 若有镜头参数则先启用；也可上传至少 5 张不同角度的棋盘格照片现场标定。
3. 自动定位内容层或拖动四角。自动结果包含接受状态、置信度、拒绝原因和画幅估计；低置信度不会静默导出。镜头启用时，定位基于去畸变代理图，四角坐标对应校正后画面。
4. 对弯曲银幕启用 Mesh，生成对称起点后拖动控制点；网格作用于透视恢复后的画面。
5. 默认使用验证过的自适应去摩尔纹；规则栅格仍明显时可选“增强去纹”。高光光晕可按
   场景预设、证据门控、增强抑制或关闭，电影原片已有的轮廓光默认保留。
6. 选择默认 Fidelity 或明确的 AI Enhanced；后者只能选择服务端已发现的本地增强模型。
7. 开始恢复后比较 Original、Geometry、Fidelity 和 AI Enhanced，再下载当前阶段 PNG。诊断区会区分实际多帧观测补回、仍未解决区域、伪影门控证据与 AI 生成细节声明。

默认上传总量上限为 256 MiB、单图上限为 8000 万像素、并发任务为 2。需要局域网访问时必须明确使用 `--host 0.0.0.0 --allow-remote`；该模式没有账号认证，只应在可信网络使用。

当前版本化接口为：`GET /api/v1/health`、`GET /api/v1/models`、`POST /api/v1/detect`、`POST /api/v1/calibrate` 和 `POST /api/v1/restore`。三个 POST 接口均使用 `multipart/form-data`，恢复结果直接返回 PNG，有限诊断位于 `X-ScreenRestore-Diagnostics` 响应头。浏览器只提交模型 ID，不能提交清单路径或命令。

## Windows 10/11 安装与运行

需要 64 位 Python 3.11。建议在 Windows 原生 PowerShell 中执行：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 install
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 run
```

等价的手工命令：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m screenrestore.app
```

打开一张初始图片：

```powershell
screenrestore-gui "D:\照片\倾斜屏幕.jpg"
```

## CLI

```powershell
screenrestore "D:\照片\输入.jpg" `
  --preset display `
  --corners auto `
  --quad-model "D:\模型\quadlocator-s.onnx" `
  --output "D:\照片\输出.png" `
  --json-diagnostics
```

`--corners auto` 采用失败关闭策略。缺少语义模型时，经典检测器只返回诊断候选，并因无法证明 content/outer 层级而拒绝无人值守继续；拒绝会返回原因，不会改用整幅图片冒充内容边界。

显式四角可使用像素或 `[0,1]` 归一化坐标，顺序会自动规范化：

```powershell
screenrestore input.jpg --corners 120,80 1810,65 1880,1010 90,1040 -o output.tiff
```

从项目处理：

```powershell
screenrestore --project work.screenrestore.json --output output.webp --quality 94
```

## QuadLocator 数据、训练与端到端几何基准

训练栈与产品依赖隔离。下面的程序合成器不读取网络素材，可用于验证嵌套画框、卡纸、画芯和屏幕边框的完整训练契约：

```bash
source .venv/bin/activate
which python
python -m pip install -r training/requirements.txt
python -m training.quadlocator.generate_synthetic --output-directory /tmp/quad-synth --count 200
python -m training.quadlocator.train --manifest /tmp/quad-synth/manifest.jsonl --output-directory /tmp/quad-run --epochs 2
python -m training.quadlocator.export_onnx --checkpoint /tmp/quad-run/best.pt --output /tmp/quadlocator-s.onnx
```

真实数据清单遵循 `datasets/schemas/geometry.schema.json`，加载时会拒绝同一 `group_id` 或 `capture_session` 跨训练、验证和测试集。正式 `e2e_auto` 基准在模型完成预测前只读取手机照片：

```bash
python -m benchmarks.geometry_e2e.run --quad-model /tmp/quadlocator-s.onnx
```

SmartDoc 等真实数据接入标准 `geometry.schema.json` JSONL 后，可将照片目录与清单一同
传入。基准先扫描并预测该目录中的照片，全部预测冻结后才读取清单里的类别与四角真值；
`test` split 默认用于评分，`group_id` 按独立 group 计入 release 最低样本数：

```bash
python -m benchmarks.geometry_e2e.run \
  --data-directory "$SCREENRESTORE_DATA_ROOT/geometry/smartdoc/frames" \
  --manifest "$SCREENRESTORE_DATA_ROOT/manifests/smartdoc.geometry.jsonl" \
  --dataset-root "$SCREENRESTORE_DATA_ROOT" \
  --quad-model /tmp/quadlocator-s.onnx
```

同一清单用于训练时也必须指定数据根，避免清单位于 `manifests/` 目录时错误地把图片
解析为其子目录：

```bash
python -m training.quadlocator.train \
  --manifest "$SCREENRESTORE_DATA_ROOT/manifests/smartdoc.geometry.jsonl" \
  --dataset-root "$SCREENRESTORE_DATA_ROOT" \
  --output-directory "$SCREENRESTORE_RUN_ROOT/geometry/smartdoc"
```

未训练的 smoke checkpoint 只验证接口，不代表模型质量。发布 gate 至少需要 100 个独立实拍 group；当前四场景仅作回归烟测。

## P1 数据训练入口

数据根的 P1 训练契约已固定：SmartDoc 的 `content_quad` 仅训练 QuadLocator；DIV2K HR
仅训练 Fidelity 的受约束恢复；无 GT 的 private 图片只可显式加入 identity 保护，不能伪造
四角或 clean target。DIV2K wild 是真实 x4 退化数据，记录在清单的 `wild_x4_images` 字段，
当前用于真实退化域差审计，不与同尺寸 Fidelity 训练混合。

最简单的 MPS 烟测入口：

```bash
export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
export SCREENRESTORE_RUN_ROOT="$HOME/screenrestore-runs"
bash scripts/train_p1.sh smoke --with-private-identity
```

`smoke` 使用有限样本验证数据、MPS、checkpoint 与 ONNX；不带参数的 `full` 才是正式全量
入口，依次训练 SmartDoc 几何、DIV2K 同尺寸 Fidelity、DIV2K x2 bicubic 超分和 DIV2K wild
x4 超分。运行结束会在 `$SCREENRESTORE_RUN_ROOT/<run>/` 生成各阶段 checkpoint、ONNX、`run.json`
和训练历史。几何 ONNX 可直接供 CLI 的 `--quad-model` 使用；恢复 ONNX 的本地模型清单、私有
无 GT 审计及全部参数说明见 [训练说明](docs/TRAINING.md)。

P2 只修复和重训 geometry，不会重跑 Fidelity 或 super-resolution。新契约增加显式
`outer_presence_logits`，旧 6-output P1 ONNX 会被运行时明确拒绝。数据准备、private 标注、
合成数据与阶段清单先独立完成，再由操作者逐阶段启动：

```bash
source .venv/bin/activate
which python
export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
export SCREENRESTORE_RUN_ROOT="$HOME/screenrestore-runs"

python scripts/prepare_p2_geometry_data.py --dataset all \
  --data-root "$SCREENRESTORE_DATA_ROOT" --met-count 1500
python scripts/label_private_geometry.py \
  --data-root "$SCREENRESTORE_DATA_ROOT" \
  --image-directory "$SCREENRESTORE_DATA_ROOT/private" \
  --output "$SCREENRESTORE_DATA_ROOT/private/geometry.annotations.jsonl"
python -m training.quadlocator.generate_synthetic \
  --output-directory "$SCREENRESTORE_DATA_ROOT/geometry/synthetic" \
  --count 24000 --size 640 --negative-ratio 0.25 \
  --content-directory "$SCREENRESTORE_DATA_ROOT/textures/met-open-access/images" \
  --content-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR" \
  --background-directory "$SCREENRESTORE_DATA_ROOT/backgrounds/coco/val2017" \
  --background-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR"
python scripts/build_p2_geometry_manifests.py --data-root "$SCREENRESTORE_DATA_ROOT"

export SCREENRESTORE_RUN_NAME="p2-geometry-w1-$(date +%Y%m%d-%H%M%S)"
export P2_DEVICE=mps
bash scripts/train_p2_geometry.sh preflight
bash scripts/train_p2_geometry.sh stage-a
bash scripts/train_p2_geometry.sh stage-b
bash scripts/train_p2_geometry.sh stage-c
bash scripts/train_p2_geometry.sh stage-d
```

合成器会在生成前固定拆分公开内容与背景纹理，同一源作品或场景只进入一个 split；private
层级不确定样本按自动拒绝语义提供 presence 负监督。

每个训练 stage 自动导出 7-output ONNX、运行 SmartDoc e2e 诊断，并生成 public validation
至少 50 张及 private validation/test 全量 overlay。Stage D 冻结权重，只读取 validation
校准置信度；test 不参与阈值或模型选择。完整 P2 说明见 [训练说明](docs/TRAINING.md)，正式训练与
验收结论见 [P2 自动几何结果](docs/P2_GEOMETRY_RESULTS.md)。

第二阶段的专项训练采用统一配对清单契约。准备真实去噪、去模糊、色彩/光照、反光、去摩尔纹
或超分数据后，先执行审计，确保配对尺寸、图像可解码性以及 `group_id` / `capture_session` 不会
跨 split 泄漏：

```bash
source .venv/bin/activate
which python
python scripts/audit_restoration_manifest.py \
  --data-root "$SCREENRESTORE_DATA_ROOT" \
  --manifest "$SCREENRESTORE_DATA_ROOT/manifests/<专项>.restoration.jsonl"
```

每个 Fidelity checkpoint 都应额外运行固定退化切片评测，再决定光度、噪声、模糊或压缩退化的
采样比例和损失权重；不得只依据混合验证集的总分调参：

```bash
python -m training.restoration.evaluate_slices \
  --checkpoint "$SCREENRESTORE_RUN_ROOT/<run>/restoration/best.pt" \
  --hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_valid_HR" \
  --output "$SCREENRESTORE_RUN_ROOT/<run>/restoration/evaluation-slices.json" \
  --samples 100 --device auto
```

DocAligner 的上游工具依赖较多，建议在独立 benchmark venv 中复现，不进入核心安装：

```bash
source .venv/bin/activate
which python
python -m venv benchmarks/geometry_e2e/.venv-docaligner
source benchmarks/geometry_e2e/.venv-docaligner/bin/activate
which python
python -m pip install -r benchmarks/geometry_e2e/requirements-docaligner.txt
python -m pip install -e .
MPLCONFIGDIR=/tmp/screenrestore-matplotlib python -m benchmarks.geometry_e2e.docaligner_baseline --model-config fastvit_sa24
```

## 测试与打包

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 test
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 package
```

或手工执行：

```powershell
.\.venv\Scripts\Activate.ps1
python -m ruff check src tests scripts
python -m pytest
python -m PyInstaller --noconfirm packaging/screenrestore.spec
```

Windows 构建结果位于 `dist\ScreenRestore\ScreenRestore.exe`。PyInstaller 必须在 Windows 原生环境运行；WSL/Linux 构建不会产生 Windows 可执行文件。

## 可选模型

核心程序不会下载模型。清单必须明确声明 `restoration` 或 `enhancement` 角色及任务；
桌面流水线可配置路径，本地网页只展示服务器允许目录中的 ID，并从当前流水线实际
调用同一后端。

Windows 一键安装官方权重与 CPU PyTorch：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin torch
```

安装官方 Real-ESRGAN ncnn-vulkan Windows 包：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin ncnn
```

所有下载都校验 SHA-256，并保存到 Git 忽略目录。详细清单字段、直接运行命令、
效果取舍和安全注意事项见 [MODEL_PLUGINS.md](MODEL_PLUGINS.md)。

## 明确限制

- 镜头畸变和弯曲银幕已经分别由相机模型与 Mesh 校正；滚动快门逐行几何、任意自由曲面和自动估计 Mesh 仍需手工控制点或后续时序模型。
- 多帧可从其他照片的真实可见像素补回瞬态反光、过曝、遮挡和局部模糊，但要求画面内容一致且仍有至少一帧清晰可见。所有输入在同一区域都已饱和、纯黑、遮挡或彻底模糊时，不存在可真实恢复的信息。
- 单图的小区域 inpainting 与学习模型属于邻域/统计推断，不等于原始数字帧。Wiener 反卷积仍是实验功能，PSF 参数不匹配会振铃。
- 联合去摩尔纹会优先处理局部长时间保持同一方向的屏幕栅格，并保护大尺度轮廓；与真实细密纹理同频同向的残留仍存在不可判定区间，自动频域陷波因此默认关闭。
- 当前 Web 多图输入按同场景连拍处理，不是互不相关图片的批量导出队列；视频解码、逐帧闪烁稳定和滚动快门时序恢复尚未实现。
- Restoration 模型用于去模糊、去噪、去摩尔纹等观测恢复先验；Enhancement 模型会生成统计意义上的细节，不能保证等于原始数字帧。两类结果不会混用同一个名称或覆盖 Fidelity。
- ScreenRestore 只把多帧中确实存在的替代像素标为“观测补回”，不会把反光抑制或 inpainting 描述为大面积真实恢复。

## Oracle 恢复基准

`测试数据/` 中的 7 组显示器照片及“电影测试二”都有对应数字原图。该协议会先用数字原图经 SIFT + MAGSAC 求出实拍中的内容四角，再测试准确几何下的后处理能力。因此它是 `oracle_restoration`，不能代表自动定位或端到端自动扫描成绩。以下命令生成逐例 Geometry、Fidelity、AI Enhanced、差异热图、JSON 指标和离线 HTML 对比页：

```bash
source .venv/bin/activate
which python
python scripts/evaluate_paired.py
```

在这套 oracle 几何基准中，Fidelity 相对 Geometry 的亮度 SSIM 为 8/8 改善、PSNR 为 7/8
改善；平均 PSNR 从 19.9299 提升到 20.5850 dB，平均亮度 SSIM 从 0.7631 提升到
0.8063，平均 ΔE 从 10.5796 降到 9.4101。测试 4 和测试 6 的亮度 SSIM 分别提升
0.0980 和 0.0863；测试 5 的 PSNR 变化为 -0.0101 dB、SSIM 仍提升 0.0041，避免为
降纹把自然图像整体磨平。“测试6”由大面积白底证据触发白场校正；“电影测试二”会额外
生成五阶段消融。结果以 `validation_outputs/paired_reference/diagnostics.json` 为准，
不用单个样例的锐度数字替代整体质量判断。

## 隐私与日志

所有图像处理均在本机完成。日志不会记录图片像素或诊断图，只记录应用/系统/依赖版本、后端、算子耗时和异常堆栈。Windows 日志位于：

```text
%LOCALAPPDATA%\ScreenRestore\logs\screenrestore.log
```

## 文档

- [架构](ARCHITECTURE.md)
- [算法与失败场景](ALGORITHMS.md)
- [模型插件](MODEL_PLUGINS.md)
- [屏摄 AI 候选核查](docs/SCREEN_AI_RESEARCH.md)
- [开发记录](docs/DEVELOPMENT.md)
- [参考项目清单](docs/REFERENCE_PROJECTS.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- [P3 实施计划](plan/p3.md)
- [P3 数据采集](docs/P3_DATA_CAPTURE_GUIDE.md)
- [P3 数据审计](docs/P3_DATA_AUDIT.md)
- [P3 正式训练命令](docs/P3_TRAINING_COMMANDS.md)
- [P3 当前结果](docs/P3_RESULTS.md)

截图统一放在 `docs/screenshots/`，便于后续 Windows 原生构建时更新，不将用户输入图片作为截图素材。
