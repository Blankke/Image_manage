# P3 本地真实数据采集与标注指南

## 采集目标

P3 真实数据用于验证自动几何、忠实恢复、摩尔纹、反射和轻度弯曲。每组应保留完整 capture session，而训练切分以 `subject_id`、`group_id`、`capture_session` 三重隔离；同一作品、卡片、屏幕内容、地点连拍或设备 burst 不得跨 split。

## 通用拍摄要求

- 保留原始照片、EXIF 方向、设备型号、镜头、焦段、分辨率和拍摄时间；原图只读。
- 每个 subject 至少拍摄正面参考、轻/中度透视、近边界、小目标、弱边、模糊、曝光变化和室内外光照。
- 第一版正样本要求完整可见、平面矩形、无大面积遮挡。partial、multi-target、曲面、严重反光和极端视角应作为明确拒绝样本。
- 同一画面若含多幅画，标为 `gallery_multi_target`、`ambiguous=true`、`in_scope=false`，不得任选一个四边形作为正样本。
- 四角始终为 content 层左上、右上、右下、左下；outer 独立标注，不通过面积猜层级。

## 几何采集

每个 subject 建议 12–20 张，覆盖：

- 0–15°、15–30° 和首版上限附近的透视；
- content 与 outer 边缘同时明显的 nested-layer；
- 暗边、浅色墙面、玻璃边、画框阴影和局部反光；
- 完整目标、near-border、partial、none、multi-target 和非平面 hard negative；
- 至少两种设备或焦段，且 test 设备/地点应与训练集独立。

标注后必须生成 overlay 人工复核。任何角点含糊、content/outer 无法确认或目标不完整的样本进入拒绝类别。

## Fidelity 与 Photometric 配对

- Fidelity clean reference 应与退化输入内容一致并完成可靠配准。
- 合成 Fidelity 时，曝光、白平衡、CCM、tone 与 illumination nuisance 必须同时施加到 input 和 target；target 只移除 resize、noise、defocus、motion、JPEG 和 ringing。
- Photometric 采集建议同一稳定场景同时拍 RAW/高质量参考、灰卡/色卡和普通手机自动曝光版本。
- 记录 reference type、配准方法和质量；配准不足的样本不进入像素监督。

## 摩尔纹采集

- 记录显示设备、像素排列、分辨率、刷新率、相机设备、快门、焦距、距离和角度。
- 同一屏幕内容保存数字原图作为 clean reference，拍摄图作为 input。
- 覆盖 LCD/OLED/LED、文字、细线、自然纹理、低频彩纹和高频栅格。
- 每组额外拍摄 clean/no-moire 负样本，用于 clean texture identity gate。

## 反射采集

- Archive 首选固定内容的多帧真实观测：改变相机/光源角度，使各帧反射区域互补。
- 保存 reference frame、其他 observed frames、reflection mask、saturated/unresolved mask 和配准质量。
- 单帧 paired reference 只有在内容、曝光和配准可核实时才参与监督。
- 完全饱和或完全遮挡区域标记 `unresolved`，不得标为真实恢复。

## 轻度弯曲采集

- 当前只收集 mild curl/bend/lift，严重卷曲、折叠、缺失和非双射形变作为拒绝样本。
- 使用带直线/网格的授权测试卡，记录正面参考、弯曲输入和可选 dense grid。
- dense grid 采用 output→input 逆映射，记录网格尺寸、坐标系和最小 Jacobian。

## 清单最小字段

恢复清单遵循 `datasets/schemas/restoration.schema.json`，至少包含：sample/task/split、subject/group/session、reference_type、alignment、artifact labels/severity、degradation trace、input/target、device、source、license 和 license_restriction。反射和几何 mask 必须与所描述图像处于同一坐标系。

## 隐私、许可和磁盘

- 私有照片只保存在 `$SCREENRESTORE_DATA_ROOT/private`，不得提交、上传或写入日志。
- 外部数据必须保存明确许可与来源；许可不清晰时只记录 BLOCKED，不下载、不训练。
- 当前数据根硬上限 30 GiB。下载前先记录压缩包、解压、保留和最终预计大小；禁止缓存在线 augmentation。

## P4 target-domain development protocol

SmartDoc validation 当前只有 3 个独立 group，只能作为烟测切片。P4 新采集集固定为
development validation，用于 checkpoint eligibility、acceptance calibrator 拟合与阈值选择；它不进入
SmartDoc test，也不替代后续 release test。

- 首批目标为 48 个独立 subject：postcard、artwork、screen/display、poster 各 12 个；poster 按
  `target_class=artwork` 标注，并用 `scene_type=poster` 保留场景切片。
- 每个 subject 使用唯一 `digital_source_id`；同一数字内容的不同打印件、屏幕、裁切或颜色版本仍视为
  同一 source family。`group_id` 按 source family 固定，禁止按单张照片随机生成。
- 每个 subject 至少采集 2 个不同 `capture_session`，每个 session 6–10 张。session 以设备、地点和
  连续拍摄时间窗共同定义；同一 burst 不得拆分。
- 正样本覆盖正视、轻/中度透视、near-border、nested layer、弱边与轻反光；另按 class 配置完整目标
  数量约 25% 的 in-scope hard negative，包括 partial、multi-target、曲面、强遮挡和极端视角。
- 全部记录固定为 `split=validation`。calibrator 内部再按 `group_id` 哈希形成 fit/selection/evaluation
  三个互斥分区；不得把照片级随机切分当成独立验证。
- 采集前登记 `digital_source_id`、`subject_id`、`capture_session`、设备、地点代号、授权范围和预计
  字节数；入库前审计这三类 ID 与既有 calibration、SmartDoc validation/test 均无交集。
- 原图存于 `$SCREENRESTORE_DATA_ROOT/private/p4-target-domain/raw`，标注清单存于同一 private 根；
  overlay 与审计产物写入外部 run 目录。仓库只保留协议和 schema，不保存照片、文件名清单或标注。
- 每次入库前执行 `du -sk "$SCREENRESTORE_DATA_ROOT"`。当前根目录接近 23 GiB，剩余预算按 30 GiB
  硬上限控制；本轮不自动下载任何数据。

启用该切片的验收条件是四类均达到至少 8 个独立 group、三类隔离 ID 完整、标注 overlay 完成人工
复核且无 split 泄漏。在达到条件前，报告应明确写为 `insufficient_target_domain_groups`，不能用图片数
替代独立 group 数。
