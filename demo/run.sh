#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Missing project Python environment. From %s, run:\n' "$ROOT" >&2
    printf '  python3 -m venv .venv\n  .venv/bin/python -m pip install -r requirements.txt\n' >&2
    exit 1
fi
cd "$ROOT"
exec "$PYTHON" demo/server.py "$@"
