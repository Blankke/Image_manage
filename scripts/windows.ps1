<#
.SYNOPSIS
ScreenRestore 的 Windows 11/10 安装、运行、测试和打包脚本。

.EXAMPLE
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 install
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 run
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 test
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 package

.DESCRIPTION
脚本要求已安装 Python 3.11，并始终使用仓库内 .venv。不会安装模型、GPU 运行时或云端组件。
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "run", "test", "package")]
    [string]$Task = "run"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$VirtualPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

Push-Location $RepositoryRoot
try {
    if ($Task -eq "install") {
        py -3.11 -m venv .venv
        & $VirtualPython -c "import sys; print(sys.executable)"
        & $VirtualPython -m pip install --upgrade pip
        & $VirtualPython -m pip install -e ".[dev]"
    }
    elseif ($Task -eq "run") {
        & $VirtualPython -c "import sys; print(sys.executable)"
        & $VirtualPython -m screenrestore.app
    }
    elseif ($Task -eq "test") {
        & $VirtualPython -c "import sys; print(sys.executable)"
        & $VirtualPython -m ruff check src tests scripts
        & $VirtualPython -m pytest
    }
    elseif ($Task -eq "package") {
        & $VirtualPython -c "import sys; print(sys.executable)"
        & $VirtualPython -m pytest
        & $VirtualPython -m PyInstaller --noconfirm packaging/screenrestore.spec
        Write-Host "构建结果：dist\ScreenRestore\ScreenRestore.exe"
    }
}
finally {
    Pop-Location
}
