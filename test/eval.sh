#!/usr/bin/env bash
set -euo pipefail
# Usage: bash test/eval.sh TEST_JSONL RUN_DIR MODE
python test/unified_eval.py --test-data "$1" --run-dir "$2" --mode "$3" --with-embedding
