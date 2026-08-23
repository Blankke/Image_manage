"""运行 ScreenRestore 的配对实拍基准并生成 JSON/HTML 报告。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/evaluate_paired.py
    python scripts/evaluate_paired.py --only 电影测试二
    python scripts/evaluate_paired.py --skip-ai

该入口复用 ``validate_paired_samples.py`` 的 ``oracle_restoration`` 实现；执行时会
显示逐样本进度条。原图会参与四角定位和评分，任何原图像素都不会进入 Geometry、
Fidelity 或 AI 输出，因此结果不能代表自动定位或端到端能力。
"""

from validate_paired_samples import main

if __name__ == "__main__":
    raise SystemExit(main())
