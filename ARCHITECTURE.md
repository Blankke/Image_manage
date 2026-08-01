# 架构

## 边界

```text
PySide6 UI / CLI / 本地 Web UI + HTTP API
        ↓
多帧观测融合（可选）→ ImageDocument + ImagePipeline + ProcessingContext
        ↓
方向 → 镜头畸变 → 平面透视 → 弯曲 Mesh → 独立经典恢复算子
        ↓
OpenCV / NumPy / SciPy / Pillow

可选：RestorationModelOperator / EnhancementModelOperator
      → ModelManifest(role + task) → External / ONNX / OpenVINO → tiled_inference
```

- `core/` 不依赖 Qt，定义只读图像文档、参数协议、算子接口、流水线、LRU 缓存、历史和取消令牌。
- `operators/` 每个文件对应独立图像步骤。算子不得原地修改输入。
- `ui/` 只协调状态、显示和线程；算法运行在 `QThreadPool` worker，worker 只通过信号返回结果。
- `web/` 提供无框架、同源的静态前端和版本化本地 HTTP API；服务层只组合核心流水线，HTTP 层负责上传上限、并发上限、安全响应头和二进制编码，不在磁盘缓存用户图片。
- `io/` 负责加载、项目和原子导出；GUI 与 CLI 不复制算法。
- `inference/` 只在实际选择后惰性导入可选运行时，核心安装不包含它们；恢复模型与
  感知增强模型是两个固定语义节点，清单必须声明 `role` 和 `task`。
- `diagnostics/` 提供直方图、频谱和日志配置。

## 图像契约

加载器保留只读 RGB `uint8` 原图；进入 `ImagePipeline` 时只转换一次，此后节点、缓存和
模型后端统一传递 `H×W×3`、`[0,1]` 的 RGB `float32`。只有 OpenCV 明确要求 8 位的
NLM、CLAHE、inpaint 边界，以及外部进程 PNG 桥和最终编码器才量化为 `uint8`。摄影
曝光与自动白平衡增益在线性光中计算，其他面向 UI 的感知色调在 sRGB 中计算。

`ImageDocument.original_rgb` 被标记为只读。代理图按最长边缓存，默认 1600 像素；导出始终从原图重新计算。

## 流水线和缓存

每个节点状态包含算子 ID、启用状态和参数字典。缓存键由源图 ID、节点序号、上游累计签名、算子版本和参数摘要组成。参数变化只淘汰当前及后续节点；缓存使用有界 LRU，默认 512 MiB。

方向、镜头畸变、平面几何、弯曲网格和输出尺寸为固定位置；中间恢复算子允许重排。撤销/重做保存有界序列化快照，不保存图像副本。

## 多帧边界

`MultiFrameFusionParameters` 和 `align_and_fuse()` 接受多张 RGB `uint8` 图，不伪装成一元流水线算子。它位于流水线之前：先用代理图估计平移/仿射/单应逆采样矩阵，再在全分辨率上对齐，以分条 float32 计算控制峰值内存。裁切、欠曝、饱和、低饱和高亮、时域离群、局部清晰度和对齐分数共同构成观测权重。

结果同时返回融合图、置信度图、从其他帧真实补回的区域和所有输入仍未解决的区域。后续单图流水线只读取融合图；Web 诊断保留上述来源语义。

## 并发

- 主线程：控件、画布、状态更新。
- 预览 worker：读取流水线快照和只读代理，复用线程安全缓存。
- 导出 worker：读取流水线快照和只读原图，不读取代理缓存。
- 取消：每个节点和长循环检查 `CancellationToken`；外部进程收到取消后先 terminate，再超时 kill。
- 旧 generation 的 worker 结果不会覆盖新参数结果。

## 项目格式

`.screenrestore.json` 当前 `format_version` 为 4，保存应用版本、相对源图路径、SHA-256、源图尺寸、镜头、四角、Mesh、比例、完整算子顺序/开关/参数、预设和模型配置。版本 4 增加方向相干的亮度/色度联合去摩尔纹参数和独立 `dehalo` 节点；模型仍由 `restoration_model` / `enhancement_model` 两个固定语义节点承载。项目不保留旧格式双解析分支。源图缺失或哈希变化会警告，但允许重新定位或继续。

## Web 安全与隐私

- 默认绑定 `127.0.0.1`；非回环地址必须显式 `--allow-remote`。
- 上传要求 `Content-Length` 和 `multipart/form-data`，限制总字节、字段数、单图像素数和同时处理任务数。
- 前端与 API 同源，响应设置 CSP、`nosniff`、no-referrer、same-origin 和 `no-store`。
- 上传字节与解码数组只存在于请求生命周期，不写临时上传目录；日志不记录字段内容、文件名或像素。
- 浏览器只能提交服务端白名单目录中发现的模型 ID，不能提交清单路径、可执行命令或
  任意算子路径；模型状态响应也不会泄露本机绝对路径。

## 扩展新算子

1. 创建继承 `ParameterModel` 的 dataclass，定义稳定范围与 `validate()`。
2. 实现 `ImageOperator`，遵守 RGB/只读输入契约并检查取消。
3. 注册到 `build_registry()`，更新安全默认顺序和相关预设。
4. 添加参数序列化、数值质量、缓存和 CLI/GUI 可达性测试。
5. 若参考第三方代码，先更新许可证清单和第三方声明。
