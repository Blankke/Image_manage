<#
.SYNOPSIS
下载并校验 ScreenRestore 的可选本地模型插件；核心应用不需要运行本脚本。

.EXAMPLE
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin torch
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin ncnn
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin all

.DESCRIPTION
torch 会安装 CPU 版 PyTorch，并下载 Real-ESRGAN general-x4v3 强/弱降噪权重与
NAFNet GoPro width32 权重。ncnn 会安装官方 Windows Vulkan 程序和随包模型。
所有文件都保存在已被 Git 忽略的 models/weights 或 plugins/realesrgan 中。
#>

param(
    [ValidateSet("torch", "ncnn", "all")]
    [string]$Plugin = "all"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$VirtualPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$WeightsDirectory = Join-Path $RepositoryRoot "models\weights"

function Get-VerifiedFile {
    param(
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$Destination,
        [Parameter(Mandatory)] [string]$Sha256
    )
    if (Test-Path $Destination) {
        $ExistingHash = (Get-FileHash -Algorithm SHA256 $Destination).Hash.ToLowerInvariant()
        if ($ExistingHash -eq $Sha256) {
            Write-Host "已存在且校验通过：$Destination"
            return
        }
    }
    Write-Host "下载：$Url"
    Invoke-WebRequest -Uri $Url -OutFile $Destination
    $ActualHash = (Get-FileHash -Algorithm SHA256 $Destination).Hash.ToLowerInvariant()
    if ($ActualHash -ne $Sha256) {
        throw "SHA256 不匹配：$Destination`n期望 $Sha256`n实际 $ActualHash"
    }
}

Push-Location $RepositoryRoot
try {
    if (-not (Test-Path $VirtualPython)) {
        throw "未找到 .venv。请先运行 scripts/windows.ps1 install"
    }
    & $VirtualPython -c "import sys; print(sys.executable)"

    if ($Plugin -in @("torch", "all")) {
        & $VirtualPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch
        New-Item -ItemType Directory -Force -Path $WeightsDirectory | Out-Null
        Get-VerifiedFile `
            -Url "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth" `
            -Destination (Join-Path $WeightsDirectory "realesr-general-x4v3.pth") `
            -Sha256 "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
        Get-VerifiedFile `
            -Url "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth" `
            -Destination (Join-Path $WeightsDirectory "realesr-general-wdn-x4v3.pth") `
            -Sha256 "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca"
        Get-VerifiedFile `
            -Url "https://drive.usercontent.google.com/download?id=1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj&export=download&confirm=t" `
            -Destination (Join-Path $WeightsDirectory "NAFNet-GoPro-width32.pth") `
            -Sha256 "19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c"
    }

    if ($Plugin -in @("ncnn", "all")) {
        $PluginDirectory = Join-Path $RepositoryRoot "plugins\realesrgan"
        New-Item -ItemType Directory -Force -Path $PluginDirectory | Out-Null
        $Archive = Join-Path ([System.IO.Path]::GetTempPath()) "ScreenRestore-realesrgan-windows.zip"
        Get-VerifiedFile `
            -Url "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip" `
            -Destination $Archive `
            -Sha256 "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"
        Expand-Archive -Path $Archive -DestinationPath $PluginDirectory -Force
        Remove-Item -LiteralPath $Archive -Force
    }
    Write-Host "可选模型安装完成；请在应用的‘可选模型恢复/超分’步骤填写 models/examples 下的清单路径。"
}
finally {
    Pop-Location
}
