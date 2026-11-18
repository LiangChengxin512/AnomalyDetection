#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
exec "${PYTHON:-python}" "$PROJECT_DIR/scripts/reproduce_all.py" "$@"
