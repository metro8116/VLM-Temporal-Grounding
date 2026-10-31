#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$repo_root/tools/train.py" --config "$repo_root/timelens/configs/qlora_a800.json" "$@"
