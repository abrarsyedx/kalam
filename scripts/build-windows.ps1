# Builds the Kalam installers on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
#
# Takes care of the Windows-specific bits:
#  - a short build folder (whisper.cpp's Vulkan build nests folders deeply and
#    Windows limits paths to 260 characters),
#  - finding the Vulkan SDK and Ninja,
#  - copying the finished installers to .\dist.
param(
    # Short folder for build output. Change it if C: is not suitable.
    [string]$TargetDir = "C:\kt"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent

# Pick up tools installed since this terminal was opened.
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')

if (-not $env:VULKAN_SDK) {
    $env:VULKAN_SDK = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
}
if (-not $env:VULKAN_SDK -or -not (Test-Path $env:VULKAN_SDK)) {
    $sdk = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if (-not $sdk) { throw "Vulkan SDK not found. Install it with: winget install KhronosGroup.VulkanSDK" }
    $env:VULKAN_SDK = $sdk.FullName
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "Ninja not found. Install it with: winget install Ninja-build.Ninja (then open a new terminal)"
}
if (-not $env:LIBCLANG_PATH -and (Test-Path 'C:\Program Files\LLVM\bin')) {
    $env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
}
$env:CARGO_TARGET_DIR = $TargetDir
Write-Host "Vulkan SDK : $env:VULKAN_SDK"
Write-Host "Build dir  : $env:CARGO_TARGET_DIR"

Push-Location $root
try {
    if (-not (Test-Path node_modules)) { npm install --no-fund --no-audit }
    npx tauri build
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit code $LASTEXITCODE)" }
    New-Item -ItemType Directory -Force dist | Out-Null
    # Only this version's installers (the build folder keeps older ones).
    $version = (Get-Content src-tauri\tauri.conf.json -Raw | ConvertFrom-Json).version
    Get-ChildItem "$TargetDir\release\bundle" -Recurse -Include "*_${version}_*.exe", "*_${version}_*.msi" | ForEach-Object {
        Copy-Item $_.FullName dist -Force
        Write-Host ("Installer: dist\{0} ({1:N0} MB)" -f $_.Name, ($_.Length / 1MB))
    }
} finally {
    Pop-Location
}
