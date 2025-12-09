#!/bin/bash
# Calls overall_balancedness.py for every program in best_programs

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEST_PROGRAMS_DIR="${SCRIPT_DIR}/best_programs"
EVALUATOR_SCRIPT="${SCRIPT_DIR}/overall_balancedness.py"

# Check if evaluator script exists
if [ ! -f "$EVALUATOR_SCRIPT" ]; then
    echo "Error: overall_balancedness.py not found at $EVALUATOR_SCRIPT" >&2
    exit 1
fi

# Check if best_programs directory exists
if [ ! -d "$BEST_PROGRAMS_DIR" ]; then
    echo "Error: best_programs directory not found at $BEST_PROGRAMS_DIR" >&2
    exit 1
fi

# Don't exit on error for individual program evaluations - continue even if one fails
set +e

# Create results directory
RESULTS_DIR="${SCRIPT_DIR}/overall_balancedness_results"
mkdir -p "$RESULTS_DIR"

# Array to store results
declare -a results
declare -a program_names

echo "=========================================="
echo "Overall Balancedness Evaluation"
echo "=========================================="
echo "Evaluating programs in: $BEST_PROGRAMS_DIR"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Find all .py files in best_programs (excluding __pycache__)
programs=("$BEST_PROGRAMS_DIR"/*.py)

if [ ${#programs[@]} -eq 0 ] || [ ! -f "${programs[0]}" ]; then
    echo "Error: No Python programs found in $BEST_PROGRAMS_DIR" >&2
    exit 1
fi

# Evaluate each program
for program in "${programs[@]}"; do
    program_name=$(basename "$program" .py)
    program_names+=("$program_name")
    
    echo "----------------------------------------"
    echo "Evaluating: $program_name"
    echo "----------------------------------------"
    
    # Run evaluation and capture output
    result_file="${RESULTS_DIR}/${program_name}_result.json"
    
    if python3 "$EVALUATOR_SCRIPT" --json "$program" > "$result_file" 2>&1; then
        # Extract overall balancedness from JSON output
        overall_score=$(python3 -c "import json, sys; data = json.load(open('$result_file')); print(data.get('overall_balancedness', 'N/A'))" 2>/dev/null || echo "N/A")
        compute_score=$(python3 -c "import json, sys; data = json.load(open('$result_file')); print(data.get('compute_balancedness', 'N/A'))" 2>/dev/null || echo "N/A")
        efficiency=$(python3 -c "import json, sys; data = json.load(open('$result_file')); print(data.get('efficiency', 'N/A'))" 2>/dev/null || echo "N/A")
        
        results+=("$overall_score|$compute_score|$efficiency")
        
        echo "✅ Success"
        echo "   Overall Balancedness: $overall_score"
        echo "   Compute Balancedness: $compute_score"
        echo "   Efficiency: $efficiency"
        echo "   Result saved to: $result_file"
    else
        echo "❌ Failed"
        results+=("ERROR|ERROR|ERROR")
        echo "   Check $result_file for error details"
    fi
    echo ""
done

# Print summary table
echo "=========================================="
echo "Summary"
echo "=========================================="
printf "%-25s %-20s %-20s %-15s\n" "Program" "Overall Balancedness" "Compute Balancedness" "Efficiency"
echo "--------------------------------------------------------------------------------------------------------"

for i in "${!program_names[@]}"; do
    IFS='|' read -r overall compute efficiency <<< "${results[$i]}"
    printf "%-25s %-20s %-20s %-15s\n" "${program_names[$i]}" "$overall" "$compute" "$efficiency"
done

echo ""
echo "Detailed results saved in: $RESULTS_DIR"
echo "=========================================="
