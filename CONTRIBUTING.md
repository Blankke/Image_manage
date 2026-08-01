# 参与开发

## 环境

使用 Python 3.11 和仓库内 `.venv`。Windows 可执行：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 install
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 test
```

Linux/WSL 验证示例：

```bash
source .venv/bin/activate
which python
python -c 'import sys; print(sys.executable)'
python -m ruff check src tests
QT_QPA_PLATFORM=offscreen python -m pytest
```

## 变更要求

- 遵守 [AGENTS.md](AGENTS.md) 的架构、RGB、线程、隐私和许可证边界。
- 公共 API 提供类型标注与 docstring；关键算法用中文说明假设和失败场景。
- 新算子必须有 dataclass 参数模型、稳定范围、注册项、预设决策和数值测试。
- 测试图片必须程序生成；不得提交用户图片、模型权重、大型数据集、外部二进制或 `.env`。真实样例和官方权重只可放入已忽略目录并提供可复现脚本。
- 修改后运行 Ruff 和受影响测试；里程碑级修改运行完整测试。
- 引用第三方代码前先核查许可证，并更新 `docs/REFERENCE_PROJECTS.md` 与 `THIRD_PARTY_NOTICES.md`；权利人单独授权必须与公开许可证分开记录。

## 提交

提交消息建议使用 `feat:`、`fix:`、`test:`、`docs:`、`refactor:` 等 Conventional Commits 前缀，说明动机、实现和验证结果。不要提交构建目录、缓存、虚拟环境或包含 GPS 的真实图片。
