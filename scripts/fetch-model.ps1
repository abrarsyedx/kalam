# Downloads the Whisper models bundled with Kalam:
#   ggml-large-v3-turbo-q5_0.bin  (~574 MB, "Accurate", default)
#   ggml-small.en-q5_1.bin        (~190 MB, "Fast")
# The build does this automatically; use this script if you prefer to fetch
# them up front or the automatic download is blocked.
$ErrorActionPreference = 'Stop'
$dir = Join-Path $PSScriptRoot '..\src-tauri\resources\models'
$base = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main'
$models = @(
    @{ File = 'ggml-large-v3-turbo-q5_0.bin'; MinMB = 500 },
    @{ File = 'ggml-small.en-q5_1.bin'; MinMB = 150 }
)
New-Item -ItemType Directory -Force $dir | Out-Null
$ProgressPreference = 'SilentlyContinue'
foreach ($m in $models) {
    $dest = Join-Path $dir $m.File
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt ($m.MinMB * 1MB))) {
        Write-Host "Already present: $($m.File)"
        continue
    }
    Write-Host "Downloading $($m.File) ..."
    Invoke-WebRequest -Uri "$base/$($m.File)" -OutFile $dest
    Write-Host ("Saved {0} ({1:N0} MB)" -f $dest, ((Get-Item $dest).Length / 1MB))
}
