# ScreenRestore 开发约束

## 产品目标与当前里程碑

- ScreenRestore 面向画作、明信片与屏摄图像的本地电子化恢复。
- 当前最高优先级是平面目标的端到端自动几何：识别正确内容层、精确四角、透视校正、可信画幅和自动拒绝。
- 第一版支持完整可见的平面矩形、普通手机焦段、轻度至中度透视和无大面积遮挡场景。
- 弯曲纸张、曲面银幕、大面积缺失、严重遮挡和极端视角属于后续能力；当前自动模式应拒绝这些输入。
- 经典滤镜、通用增强和更多 UI 功能不得绕过几何 release gate 成为主路径。

## 归档语义

- `content_quad` 表示应被电子化的画芯、卡片图像区域或屏幕显示区域。
- `outer_quad` 表示画框、卡片实体或屏幕外框。两者必须独立建模和标注，不得用面积或矩形度猜测层级。
- 自动定位必须返回 `accepted` 或 `rejected`。证据不足时拒绝，禁止为了保持流程运行而强制输出四角。
- `none`、目标不完整、层级不确定和低置信度均属于正常拒绝结果。
- 传统轮廓候选可用于诊断和无模型 fallback，但它没有内容层语义，不得单独作为无人值守接受依据。
- 自动接受阈值以“高置信度错误率”优先，覆盖率为次要指标。

## Archive 与 Enhanced

- `Archive` 只允许几何校正、受约束的光度校正、由输入观测支持的恢复和多帧真实像素融合。
- `Enhanced` 可在 Archive 之上使用反射分离、inpainting、超分辨率或生成先验。
- 每个输出应能区分 `observed`、`recovered_observation`、`estimated`、`generated` 和 `unresolved`。
- 单张输入中已经饱和或完全遮挡的区域不能标记为真实恢复；生成或估计内容必须保存像素级来源蒙版。
- 原图只读。处理结果、预览代理、来源蒙版和导出图像必须是独立对象，任何算子不得原地修改输入。

## 架构边界

- `src/screenrestore/core` 只定义文档、参数、算子协议、流水线、缓存和取消机制，不依赖 Qt、模型运行时或具体图像算法。
- `src/screenrestore/geometry` 是自动定位、内容/外框语义、置信度、边缘精修、画幅估计和透视变换的唯一权威实现。
- `src/screenrestore/operators` 提供可序列化流水线算子；几何算子调用 `geometry`，不得复制几何数学或自动检测逻辑。
- `src/screenrestore/restoration` 放受约束的光度与低层恢复算法，不承担流水线编排。
- `src/screenrestore/provenance` 统一管理像素来源与 Archive/Enhanced 报告。
- `src/screenrestore/io` 负责图像、元数据和项目文件；GUI、Web 与 CLI 必须调用同一定位服务和同一流水线。
- `src/screenrestore/inference` 仅放可选模型后端；核心模块不得导入可选训练框架或推理运行时。
- `src/screenrestore/semantic` 可做场景分析和兼容适配，不再拥有另一套候选生成、四角排序或接受策略。
- `src/screenrestore/ui` 与 `src/screenrestore/web` 只负责交互、任务调度和展示，图像算法不得写在 QWidget、QObject 或路由函数中。
- `training` 与产品运行时隔离；训练代码可以依赖 PyTorch，产品核心依赖中不得强制加入 PyTorch。
- `datasets` 只提交 schema、清单范例和生成规则，不提交私有照片、大型数据集或外部权重。
- `benchmarks` 按协议组织可复现评估，不在产品模块中嵌入参考图注册逻辑。

## 自动几何主链

正式自动路径固定为：

1. 读取 EXIF 方向并按相机 profile 去镜头畸变。
2. QuadLocator 输出 content/outer 角点热图、内容 mask、boundary、presence、类别和置信度。
3. 将粗角点投回原始分辨率，在窄带内融合边界图与梯度并拟合四条直线。
4. 由统一置信度策略检查目标存在性、类别、内容层、几何、边界支持和异常分布。
5. 高置信度结果自动继续，低置信度结果明确拒绝。
6. 透视校正后输出画幅估计值、来源和置信度。

- 模型不得直接回归八个坐标作为唯一监督；角点热图、内容 mask、边界和 presence 是必需输出。
- 高分辨率精修只负责像素落点，不负责决定目标语义层。
- 精修必须有限位和接受门，证据不足时保留粗角点；不得搜索到相邻画框后仍报告成功。
- 未知相机内参下的 AUTO 画幅只能称为估计；UI 和 API 必须暴露 ratio、source 与 confidence。

## 光度与恢复约束

- 普通画作和明信片优先使用受约束的摄影校正：白平衡增益、3×3 色彩矩阵、平滑 illumination field、全局曝光和单调 tone curve。
- PhotoCalib 类模型输出受限参数，不直接生成整张 RGB 图。
- 轻量像素恢复采用有界 residual，并加入 clean identity 约束，避免改变笔触、胶片颗粒、纸张纹理、文字和人脸结构。
- ARTWORK 默认以色彩忠实为先，CLAHE、Dehalo、Demoire 关闭，锐化保持轻度。
- CINEMA 的亮度与色度应拆分处理，可信暗场可估计加性黑位；默认保护暗部 chroma，不自动中和电影固有色调。
- DISPLAY/LED 的 banding 与 moire 分别评估；专用模型通过 clean-texture gate 后才能替代经典 fallback。
- GLOSSY_ARTWORK 的 Archive 主路线是多帧对齐、反光概率和真实观测融合；单图分离属于 Enhanced。
- 损伤修复必须先输出 damage mask；大面积补全始终标记 `generated`。

## 图像约定

- 模块间统一传递 `numpy.ndarray` RGB 图像，形状为 `H×W×3`。
- 加载器输出 `uint8`；核心流水线内部使用 `[0, 1]` 的 `float32`，输出前裁剪并按导出格式转换。
- 仅在 OpenCV 的 BGR/LAB/YCrCb/HSV API 边界显式转换，并在变量名或注释中标明颜色空间。
- Alpha 通道不进入恢复流水线；加载器单独保留元数据，导出器按格式能力处理。
- 边界 mask、反光 mask、damage mask 和 provenance mask 必须与它们所描述的几何坐标系一致。

## 数据与标注

- 几何清单遵循 `datasets/schemas/geometry.schema.json`。
- 真实样本至少记录 `content_quad`、可选 `outer_quad`、`target_class`、`split`、`group_id`、`capture_session`、可见性、遮挡和反光等级。
- 四角顺序统一为左上、右上、右下、左下，坐标归一化到 `[0, 1]`。
- 合成数据重点覆盖 content、卡纸、内框、外框、墙面和屏幕边框构成的嵌套矩形，以及阴影、反光、曝光、色温、噪声、模糊、JPEG 和暗角。
- 同一作品、地点、设备连拍或 capture session 不得跨 train/validation/test；数据加载前应检查 group 泄漏。
- hard negatives 必须包含无目标、边界不完整、外框比画芯更显眼、目标超出画面和非平面目标。
- 测试图片由程序合成或来自已明确授权的本地测试集，不引入来源不明、隐私敏感或受版权限制的外部样例。
- 分析 JSONL、CSV、XLSX 时只打印字段名、长度和截断内容，禁止将整段图像关联文本输出到日志。

## 评估协议

- `oracle_restoration` 允许 clean reference 参与注册和求四角，只衡量“已知准确几何后的恢复能力”。
- `e2e_auto` 在所有预测输出完成前只能读取手机照片；clean reference 或人工角点只能在预测冻结后用于打分。
- README、release note 和 UI 不得用 oracle 指标代表自动定位或自动扫描能力。
- 当前四张实拍只作为 smoke regression，不能支持发布级可靠性结论。
- release 几何集至少包含 100 个独立 group，推荐 2000 张以上认真标注的真实训练图和独立设备/场地测试集。
- 首版 release gate：自动接受正确率 ≥99%、in-scope 接受覆盖率 ≥90%、错选层级 <0.5%、角点 NCE P95 ≤1%、Quad IoU 中位数 ≥0.97、低分位 ≥0.93。
- 去摩尔纹需同时验证周期能量抑制和 clean texture 保留；反光需分别评估 reflection ROI、clean ROI、observed recovery 与 unresolved。
- 几何与恢复测试必须验证数值质量，不能只验证“不崩溃”。

## UI 与线程

- Qt 主线程只更新界面，不执行全分辨率定位、恢复、导出或外部模型进程。
- 长任务使用 `QThreadPool` worker；worker 通过信号返回不可变结果，禁止直接访问 QWidget。
- 参数预览使用 debounce 和代理图；关闭窗口前取消并等待任务安全退出。
- CLI、Web、GUI 对同一输入应得到一致的 accepted/rejected、角点、画幅和 provenance 语义。
- 自动拒绝要展示原因和候选预览；拒绝结果不得静默回退为全图或任意矩形并导出。

## 依赖、训练与部署

- Python 固定为 3.11，产品依赖由 `pyproject.toml` 管理。
- 执行 Python、pip、pytest、ruff、训练或导出前必须激活项目 `.venv`，并用 `which python` 或 `sys.executable` 确认解释器。
- 默认无模型、无独显、无网络时必须可启动经典流水线；缺少 QuadLocator 权重时自动模式可拒绝，不得冒充训练模型结果。
- 训练依赖使用 `training/*/requirements.txt` 或 benchmark 专用 requirements，不进入默认安装。
- 当前非 Mac 开发机可完成 CPU 训练烟测、ONNX 导出和 ONNX Runtime 验证；MPS/Core ML 性能与数值一致性在 Mac mini 上完成。
- 面向 Apple 部署的模型优先使用常规 convolution、activation、normalization、upsample 和矩阵运算，避免自定义 CUDA 算子。
- 模型权重、外部二进制、下载缓存、真实私有数据和训练 run 不提交仓库。
- 图像内容不得写日志、上传网络、发送遥测或作为未授权测试夹具提交。

## 第三方与许可证

- 引入、下载、改写或复制第三方代码前必须核查许可证，并更新 `THIRD_PARTY_NOTICES.md` 与 `docs/REFERENCE_PROJECTS.md`。
- 可选依赖、参考代码、数据集和预训练权重分别核查；代码许可证不自动覆盖数据与权重。
- 无明确开源许可证的仓库只能用于理解论文或产品思路，禁止复制代码、配置、权重和素材。
- DocAligner 仅作为独立 Apache-2.0 benchmark；其预测不具备本项目要求的 content/outer 层级与接受置信度。
- UHDM/ESDNet 可进入后续屏摄 benchmark；Moiré Zero 在未确认明确许可证前只记录研究思路。
- 不得把本地下载的字体、模型、参考仓库或少量验证数据提交到当前项目。

## 代码与测试

- 新代码用中文注释解释关键流程、物理假设、坐标系和拒绝原因；新脚本文件开头提供用途与可复现命令。
- 新算子必须有可序列化参数模型、数值测试和流水线集成测试。
- 修改后先跑受影响测试和静态检查；每个里程碑结束运行完整测试。
- CLI、中文路径、取消、缓存失效、项目文件、Web API、来源蒙版和 benchmark 协议必须有回归测试。
- 测试命令需要为状态目录设置可写位置时，使用 `/tmp` 下的 `XDG_STATE_HOME`，不得修改用户全局目录。
- 不为旧字段、旧路由或旧行为保留双套兼容逻辑；契约升级时同步修改全部调用方、测试和文档，并删除旧入口。
- 未经用户明确要求不得提交 commit；任何提交都不得包含 `.env`、模型权重、私有数据或下载缓存。
