# -*- mode: python ; coding: utf-8 -*-
# 使用范例（Windows PowerShell）：pyinstaller --noconfirm packaging/screenrestore.spec
# 说明：模型和外部程序不打入核心包；用户可在发布目录 models/plugins 中自行安装。

analysis = Analysis(
    ["../src/screenrestore/app.py"],
    pathex=["../src"],
    binaries=[],
    datas=[
        ("../models/examples", "models/examples"),
        ("../THIRD_PARTY_NOTICES.md", "."),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "onnxruntime", "openvino"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="ScreenRestore",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ScreenRestore",
)
