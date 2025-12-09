# TAPE: Topologically Aware Placement of Experts
By Emmanuel Rassou, Tomas Gonzalez & Dylan Bruckner

*This is a forked repo of OpenEvolve.*

## Overview

TAPE (Topologically Aware Placement of Experts) is an evolutionary approach to optimizing Expert Parallelism Load Balancing (EPLB) algorithms for Mixture-of-Experts (MoE) models. This repository contains the code and configuration needed to reproduce the experimental results.

## Setup

### Prerequisites

1. **Python Environment**: Python 3.8+ with pip/uv package manager
2. **API Key**: Google Gemini API key for LLM-based code evolution
3. **PyTorch**: Required for workload simulation and evaluation

### Installation Steps

1. **Set up API Key**:
   ```bash
   export OPENAI_API_KEY="your-gemini-api-key-here"
   ```
   Note: Despite the variable name `OPENAI_API_KEY`, this is used for the Gemini API (which uses OpenAI-compatible endpoints).

2. **Install PyTorch**:
   ```bash
   uv pip install torch
   ```
   Or using pip:
   ```bash
   pip install torch
   ```

3. **Install OpenEvolve** (if not already installed):
   ```bash
   cd openevolve
   pip install -e .
   ```

4. **Download Workload Data**:
   ```bash
   cd examples/eplb
   wget https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json
   mkdir -p data
   mv expert-load.json data/expert-load.json
   ```

For additional setup instructions, see `/examples/eplb/README.md`.

## Evolution Runs

The evolution process uses OpenEvolve to iteratively improve the EPLB algorithm through LLM-guided code mutations. We provide multiple TAPE variants that optimize different aspects of the load balancing problem.

### TAPE Versions

The repository includes several TAPE evaluator versions, each optimizing different combinations of metrics:

- **TAPE Version 0** (Baseline): Uses the original `evaluator.py` from "Barbarians at the Gate"
  - Optimizes: Balancedness score + Speed score
  
- **TAPE Version 2**: Topological awareness with communication cost
  - Optimizes: Balancedness score + Communication score
  - Focus: Minimize GPU-to-GPU communication during token routing
  
- **TAPE Version 3**: Adds execution speed optimization
  - Optimizes: Balancedness score + Communication score + Speed score
  - Focus: Balance load, minimize communication, and improve algorithm execution time
  
- **TAPE Version 4**: Full optimization including weight copying costs
  - Optimizes: Balancedness score + Communication score + Speed score + Weight copy score
  - Focus: Complete optimization including cost of copying expert weights between GPUs

### Running Evolution

**Important**: Before running, set the `TAPE_VERSION` parameter in `examples/eplb/TAPE_evaluator.py` to match your chosen version (2, 3, or 4). For version 0, use the separate `evaluator.py` file.

```bash
cd openevolve

# Set which TAPE version to use (0, 2, 3, or 4)
TAPE_EVALUATOR=4

# Configure paths based on version
if [ "$TAPE_EVALUATOR" = "4" ]; then
  EVALUATOR_FILE="examples/eplb/TAPE_evaluator.py"
  CONFIG_FILE="examples/eplb/TAPE4_config.yaml"
  OUTPUT_DIR="examples/eplb/output_TAPE4"
elif [ "$TAPE_EVALUATOR" = "3" ]; then
  EVALUATOR_FILE="examples/eplb/TAPE_evaluator.py"
  CONFIG_FILE="examples/eplb/TAPE3_config.yaml"
  OUTPUT_DIR="examples/eplb/output_TAPE3"
elif [ "$TAPE_EVALUATOR" = "2" ]; then
  EVALUATOR_FILE="examples/eplb/TAPE_evaluator.py"
  CONFIG_FILE="examples/eplb/TAPE2_config.yaml"
  OUTPUT_DIR="examples/eplb/output_TAPE2"
elif [ "$TAPE_EVALUATOR" = "0" ]; then
  EVALUATOR_FILE="examples/eplb/evaluator.py"
  CONFIG_FILE="examples/eplb/config.yaml"
  OUTPUT_DIR="examples/eplb/output"
fi

# Run evolution
python -m openevolve.cli \
  examples/eplb/initial_program.py \
  $EVALUATOR_FILE \
  --config $CONFIG_FILE \
  --iterations 1000 \
  --output $OUTPUT_DIR
```

### Configuration Parameters

Key parameters in the config YAML files:

- **`max_iterations`**: Number of evolution iterations (default: 1000)
- **`checkpoint_interval`**: How often to save checkpoints (default: 50)
- **`llm.primary_model`**: Primary LLM model (default: "gemini-2.5-flash")
- **`llm.secondary_model`**: Secondary LLM model for diversity
- **`database.population_size`**: Number of programs in the population (default: 1000)
- **`database.archive_size`**: Size of the elite archive (default: 100)
- **`evaluator.parallel_evaluations`**: Number of parallel evaluations (default: 4)
- **`evaluator.timeout`**: Timeout per evaluation in seconds (default: 60)

### Resuming from Checkpoint

To resume a previous evolution run:

```bash
python -m openevolve.cli \
  examples/eplb/initial_program.py \
  $EVALUATOR_FILE \
  --config $CONFIG_FILE \
  --checkpoint $OUTPUT_DIR/checkpoints/checkpoint_50 \
  --iterations 1000 \
  --output $OUTPUT_DIR
```

### Output Structure

After evolution completes, the output directory contains:

- **`best/`**: Best evolved program(s) found during evolution
- **`checkpoints/`**: Checkpoint directories for each checkpoint interval
- **`logs/`**: Evolution logs and metrics
- **`database/`**: Population database state

## Post-Training Evaluation

After evolution, you can evaluate the evolved programs using two different evaluation scripts.

### Individual Metrics Evaluation

The `openevolve_evaluate.sh` script evaluates a single program and reports individual metrics (balancedness, speed, communication, etc.).

**Usage**:
```bash
# Evaluate with standard evaluator
./openevolve/examples/eplb/openevolve_evaluate.sh \
  openevolve/examples/eplb/output_TAPE4/best/best_program.py

# Evaluate with TAPE evaluator (for TAPE versions 2, 3, 4)
./openevolve/examples/eplb/openevolve_evaluate.sh \
  openevolve/examples/eplb/output_TAPE4/best/best_program.py \
  --tape
```

**Output**: The script prints:
- Balancedness Score: How evenly load is distributed (higher is better, max 1.0)
- Speed Score: Algorithm execution speed (higher is better)
- Communication Score: Communication cost efficiency (higher is better, for TAPE versions)
- Weight Copy Score: Weight copying cost efficiency (higher is better, for TAPE version 4)
- Combined Score: Weighted combination of all metrics
- Average Raw Time: Average execution time in seconds

### End-to-End Balancedness Evaluation

The `overall_evaluate.sh` script evaluates multiple programs and computes the **overall balancedness** metric, which simulates the actual execution timeline on GPUs accounting for both computation and communication-induced idle time.

**Usage**:
```bash
# Evaluate all programs in best_programs/ directory
./openevolve/examples/eplb/overall_evaluate.sh
```

**Prerequisites**: 
- Place evolved program files in `examples/eplb/best_programs/` directory
- Each program should be a `.py` file containing the `rebalance_experts` function

**Output**: 
- Creates `overall_balancedness_results/` directory with detailed JSON results for each program
- Prints a summary table with:
  - Overall Balancedness: Elapsed time balancing across GPUs (higher is better, max 1.0)
  - Compute Balancedness: Computation load balancing (higher is better, max 1.0)
  - Efficiency: Fraction of time spent on computation vs. communication (higher is better)

**Understanding Overall Balancedness**:

The overall balancedness metric accounts for:
- **Computation time**: Time spent processing tokens on each GPU
- **Communication time**: Time spent routing tokens between GPUs
- **Idle time**: Time GPUs wait for synchronization barriers
- **Elapsed time**: Total time per GPU = compute + idle time

A score of 1.0 means all GPUs finish at exactly the same time (perfect balance). Lower scores indicate some GPUs finish earlier and remain idle while waiting for others.

For detailed documentation on the overall balancedness metric, see `examples/eplb/overall_balancedness.md`.

### Example Evaluation Workflow

```bash
# 1. Copy best programs to evaluation directory
cp openevolve/examples/eplb/output_TAPE4/best/*.py \
   openevolve/examples/eplb/best_programs/

# 2. Evaluate individual metrics for a specific program
./openevolve/examples/eplb/openevolve_evaluate.sh \
  openevolve/examples/eplb/best_programs/best_program.py \
  --tape

# 3. Evaluate overall balancedness for all programs
./openevolve/examples/eplb/overall_evaluate.sh

# 4. Check results
ls openevolve/examples/eplb/overall_balancedness_results/
cat openevolve/examples/eplb/overall_balancedness_results/best_program_result.json
```

## Reproducibility Notes

### Hardware Requirements

- **GPUs**: Not required for evaluation (uses simulated workloads)
- **CPU**: Multi-core recommended for parallel evaluations
- **Memory**: Sufficient RAM for PyTorch tensor operations (workloads can be large)
- **Storage**: Space for checkpoints and logs (can grow large over 1000 iterations)

### Software Versions

- Python 3.8+
- PyTorch (version compatible with your system)
- OpenEvolve framework

### Random Seeds

The evolution process uses LLM-based mutations, which introduces non-determinism. For reproducible results:
- Use the same API key and model versions
- Set random seeds in the config if supported
- Note that LLM responses may vary between runs

### Expected Runtime

- **Evolution**: ~1000 iterations can take several hours to days depending on:
  - Number of parallel evaluations
  - LLM API response times
  - Evaluation timeout settings
- **Post-evaluation**: Individual program evaluation takes seconds to minutes per program

## Troubleshooting

### Common Issues

1. **Missing workload file**: Ensure `data/expert-load.json` exists in `examples/eplb/`
2. **API key errors**: Verify `OPENAI_API_KEY` environment variable is set correctly
3. **Import errors**: Ensure you're running from the `openevolve/` directory or have installed the package
4. **TAPE_VERSION mismatch**: Ensure `TAPE_VERSION` in `TAPE_evaluator.py` matches your chosen version

### Getting Help

- Check `examples/eplb/README.md` for additional setup instructions
- Review evaluation logs in `output_*/logs/` directories
- Check checkpoint files for intermediate results










