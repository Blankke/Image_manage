# 架构

## 边界

```text
PySide6 UI / CLI
        ↓
ImageDocument + ImagePipeline + ProcessingContext
        ↓
独立经典算子（RGB uint8 边界，内部 float32）
        ↓
OpenCV / NumPy / SciPy / Pillow

可选：ModelPluginOperator → ModelManifest → External / ONNX / OpenVINO → tiled_inference
```

- `core/` 不依赖 Qt，定义只读图像文档、参数协议、算子接口、流水线、LRU 缓存、历史和取消令牌。
- `operators/` 每个文件对应独立图像步骤。算子不得原地修改输入。
- `ui/` 只协调状态、显示和线程；算法运行在 `QThreadPool` worker，worker 只通过信号返回结果。
- `io/` 负责加载、项目和原子导出；GUI 与 CLI 不复制算法。
- `inference/` 只在实际选择后惰性导入可选运行时，核心安装不包含它们；默认禁用的
  `ModelPluginOperator` 让 GUI、项目和 CLI 真正调用同一模型后端。
- `diagnostics/` 提供直方图、频谱和日志配置。

## 图像契约

模块间统一传递 `H×W×3` 的 RGB `uint8`。需要色调数学时，算子显式转为 `[0,1] float32`，处理后裁剪、去除 NaN/Inf 并转回 `uint8`。OpenCV BGR、LAB、YCrCb 和 HSV 只在调用边界显式转换。

`ImageDocument.original_rgb` 被标记为只读。代理图按最长边缓存，默认 1600 像素；导出始终从原图重新计算。

## 流水线和缓存

每个节点状态包含算子 ID、启用状态和参数字典。缓存键由源图 ID、节点序号、上游累计签名、算子版本和参数摘要组成。参数变化只淘汰当前及后续节点；缓存使用有界 LRU，默认 512 MiB。

方向、几何和输出尺寸为固定位置；中间恢复算子允许重排。撤销/重做保存有界序列化快照，不保存图像副本。

## 并发

- 主线程：控件、画布、状态更新。
- 预览 worker：读取流水线快照和只读代理，复用线程安全缓存。
- 导出 worker：读取流水线快照和只读原图，不读取代理缓存。
- 取消：每个节点和长循环检查 `CancellationToken`；外部进程收到取消后先 terminate，再超时 kill。
- 旧 generation 的 worker 结果不会覆盖新参数结果。

## 项目格式

`.screenrestore.json` 当前 `format_version` 为 1，保存应用版本、相对源图路径、SHA-256、源图尺寸、四角、比例、完整算子顺序/开关/参数、预设和模型配置。源图缺失或哈希变化会警告，但允许重新定位或继续。

## 扩展新算子

1. 创建继承 `ParameterModel` 的 dataclass，定义稳定范围与 `validate()`。
2. 实现 `ImageOperator`，遵守 RGB/只读输入契约并检查取消。
3. 注册到 `build_registry()`，更新安全默认顺序和相关预设。
4. 添加参数序列化、数值质量、缓存和 CLI/GUI 可达性测试。
5. 若参考第三方代码，先更新许可证清单和第三方声明。
