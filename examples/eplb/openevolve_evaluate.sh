#!/bin/bash
# Script to evaluate a program.py file using either evaluator.py or TAPE_evaluator.py
#
# Usage:
#   ./evaluate_program.sh <program.py> [--tape]
#
# Examples:
#   ./openevolve_evaluate.sh openevolve/examples/eplb/output/best/best_program.py
#   ./openevolve_evaluate.sh openevolve/examples/eplb/output_TAPE/best/best_program.py --tape

set -e

# Check if program file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <program.py> [--tape]"
    echo ""
    echo "Arguments:"
    echo "  program.py    Path to the program file to evaluate"
    echo "  --tape        Use TAPE_evaluator.py instead of evaluator.py (optional)"
    exit 1
fi

PROGRAM_FILE="$1"
USE_TAPE=false

# Check for --tape flag
if [ "$2" = "--tape" ]; then
    USE_TAPE=true
fi

# Check if program file exists
if [ ! -f "$PROGRAM_FILE" ]; then
    echo "Error: Program file not found: $PROGRAM_FILE"
    exit 1
fi

# Get absolute path to program file
PROGRAM_FILE=$(realpath "$PROGRAM_FILE")

# Get the workspace root directory (parent of openevolve)
WORKSPACE_ROOT=$(cd "$(dirname "$0")" && pwd)

# Change to openevolve directory to avoid namespace package conflict
cd "$WORKSPACE_ROOT/openevolve"

# Determine which evaluator to use
if [ "$USE_TAPE" = "true" ]; then
    EVALUATOR_FILE="examples/eplb/TAPE_evaluator.py"
    echo "Using TAPE evaluator..."
else
    EVALUATOR_FILE="examples/eplb/evaluator.py"
    echo "Using standard evaluator..."
fi

# Check if evaluator file exists
if [ ! -f "$EVALUATOR_FILE" ]; then
    echo "Error: Evaluator file not found: $EVALUATOR_FILE"
    exit 1
fi

# Get absolute path to evaluator
EVALUATOR_FILE=$(realpath "$EVALUATOR_FILE")

# Run evaluation using Python
python3 << EOF
import sys
import json
import os

# Add the openevolve directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath("$EVALUATOR_FILE")))

# Import the evaluator module
import importlib.util
spec = importlib.util.spec_from_file_location("evaluator", "$EVALUATOR_FILE")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

# Run evaluation
print(f"Evaluating program: $PROGRAM_FILE")
print("=" * 60)
result = evaluator.evaluate("$PROGRAM_FILE")
print("=" * 60)
print("\nEvaluation Results:")
print(json.dumps(result, indent=2))
print("=" * 60)

# Print summary
if "error" in result and result["error"]:
    print(f"\n❌ Error: {result['error']}")
else:
    print("\n✅ Evaluation completed successfully!")
    print(f"   Balancedness Score: {result.get('balancedness_score', 'N/A'):.6f}")
    print(f"   Speed Score:        {result.get('speed_score', 'N/A'):.6f}")
    if 'communication_score' in result:
        print(f"   Communication Score: {result.get('communication_score', 'N/A'):.6f}")
    if 'weight_copy_score' in result:
        print(f"   Weight Copy Score:   {result.get('weight_copy_score', 'N/A'):.6f}")
    print(f"   Combined Score:     {result.get('combined_score', 'N/A'):.6f}")
    if 'avg_raw_time' in result:
        print(f"   Avg Raw Time:        {result.get('avg_raw_time', 'N/A'):.6f} seconds")
EOF

