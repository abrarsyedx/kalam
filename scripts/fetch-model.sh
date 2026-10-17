#!/usr/bin/env bash
# Downloads the Whisper models bundled with Kalam:
#   ggml-large-v3-turbo-q5_0.bin  (~574 MB, "Accurate", default)
#   ggml-small.en-q5_1.bin        (~190 MB, "Fast")
# The build does this automatically; use this script if you prefer to fetch
# them up front or the automatic download is blocked.
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)/src-tauri/resources/models"
base="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
mkdir -p "$dir"
for spec in "ggml-large-v3-turbo-q5_0.bin:500000000" "ggml-small.en-q5_1.bin:150000000"; do
  file="${spec%%:*}"; min="${spec##*:}"
  if [ -f "$dir/$file" ] && [ "$(wc -c < "$dir/$file")" -gt "$min" ]; then
    echo "Already present: $file"
    continue
  fi
  echo "Downloading $file ..."
  curl -L --fail --retry 3 -o "$dir/$file" "$base/$file"
done
