#!/usr/bin/env bash
# ==============================================================================
# CS328 ITDS: Interactive Molecular Retrieval Demo Launcher
# Author: Laxmidhar Panda (Roll No: 24110185)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check Python environment
if [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
else
    echo "Error: Python 3 executable not found!"
    exit 1
fi

echo "======================================================================"
echo "  CS328 ITDS: Interactive Molecular Retrieval Demo Launcher"
echo "  Author: Laxmidhar Panda (Roll No: 24110185)"
echo "  IIT Gandhinagar | Semester V (Autumn 2026)"
echo "======================================================================"
echo "Using Python: $PYTHON_EXEC"

if [ "$1" == "--cli" ] || [ "$1" == "-c" ]; then
    echo "Starting Live Interactive Terminal CLI Mode..."
    echo "======================================================================"
    echo ""
    exec "$PYTHON_EXEC" "$PROJECT_ROOT/demo/cli_demo.py"
fi

echo "Starting local web demo server on http://127.0.0.1:8000 ..."
echo ""
echo "  👉 Open your browser to: http://127.0.0.1:8000"
echo "  👉 Or run terminal CLI:   ./demo/run_demo.sh --cli"
echo ""
echo "Press Ctrl+C to terminate the server when finished."
echo "======================================================================"
echo ""

exec "$PYTHON_EXEC" "$PROJECT_ROOT/demo/app.py"
