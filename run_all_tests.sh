#!/usr/bin/env bash
set -e

echo "================================================================="
echo "  AI Field Developer / Solutions Engineer Assessment Suite"
echo "  Running all 4 Tasks with Pytest"
echo "================================================================="

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=.
pytest -v --durations=10

echo ""
echo "================================================================="
echo "  ALL 4 TASKS PASSED SUCCESSFULLY (35/35 Tests Verified)"
echo "================================================================="
