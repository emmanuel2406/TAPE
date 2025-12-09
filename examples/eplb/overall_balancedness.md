# Overall Balancedness Evaluation

## Overview

The `overall_balancedness.py` script evaluates the **overall balancedness** of an expert parallelism load balancer by simulating the actual execution timeline on GPUs. This metric accounts for both computation time and idle time due to communication dependencies, providing a comprehensive measure of how evenly GPUs are utilized over time.

## Definition

**Overall balancedness** is defined as the balancing of elapsed time among GPUs, where:

```
elapsed_time = GPU compute time + idle time due to waiting on communication
```

This metric combines:
- **Computation balancing**: How evenly computation load is distributed across GPUs
- **Communication-induced idle time balancing**: How evenly GPUs experience idle time from waiting on communication

## Mathematical Formulation

The overall balancedness score is calculated as:

```
overall_balancedness = avg_elapsed_time / max_elapsed_time
```

Where:
- `avg_elapsed_time`: Average elapsed time across all GPUs
- `max_elapsed_time`: Maximum elapsed time across all GPUs

**Interpretation:**
- `1.0`: Perfect balance (all GPUs have identical elapsed times)
- `< 1.0`: Imbalanced (some GPUs finish significantly earlier than others)
- Lower values indicate worse balancing

## Simulation Model

The script simulates the execution timeline for MoE (Mixture of Experts) inference with data parallelism:

### 1. Per-Layer Execution

For each MoE layer:

1. **Compute Time Calculation**
   - Distributes logical expert load to physical expert replicas
   - Calculates per-GPU compute load based on expert placement
   - Compute time = `compute_load × COMPUTE_TIME_PER_TOKEN`

2. **Communication Time Calculation**
   - Models token routing from source GPUs to expert replicas on target GPUs
   - Accounts for intra-node (NVLink) vs inter-node (network) communication costs
   - Communication time depends on:
     - Number of tokens routed
     - Network distance (same GPU, same node, different node)
     - Communication cost constants (`INTRANODE_COMM`, `INTERNODE_COMM`)

3. **Elapsed Time Calculation**
   - `elapsed_time = max(compute_time, communication_time)`
   - Accounts for synchronization: all GPUs wait for the slowest GPU
   - Models the barrier synchronization between layers

4. **Idle Time Calculation**
   - `idle_time = elapsed_time - compute_time`
   - Represents time spent waiting on communication or synchronization

### 2. Accumulation Across Layers

- Compute times, idle times, and elapsed times are accumulated across all layers
- The final per-GPU elapsed time is the sum of all layer elapsed times

### 3. Overall Balancedness

- Calculates average and maximum elapsed times across all GPUs
- Overall balancedness = `avg_elapsed_time / max_elapsed_time`

## Usage

### Basic Usage

```bash
python openevolve/examples/eplb/overall_balancedness.py \
    openevolve/examples/eplb/output/best/best_program.py
```

### Options

- `--quiet` / `-q`: Suppress verbose output
- `--json`: Output results as JSON format

### Example Output

```
Overall Balancedness: 0.854321
  (Compute-only balancedness: 0.912345)
  (Efficiency: 0.782156)
  (Avg total elapsed time: 1234.56)
  (Evaluated on 10 workload pairs)

Detailed per-workload scores:
  Workload 1: 0.852341
  Workload 2: 0.856789
  ...
```

## Research Soundness

### Model Assumptions

1. **Data Parallelism**: Tokens are uniformly distributed across all GPUs
2. **Synchronous Execution**: All GPUs synchronize between layers (barrier synchronization)
3. **Communication Overlap**: Communication can overlap with computation, but synchronization requires waiting for the slowest operation
4. **Linear Scaling**: Compute time scales linearly with token count
5. **Network Model**: Communication cost is proportional to token count and network distance

### Limitations

1. **Simplified Communication Model**: 
   - Assumes all communication happens in parallel
   - Does not model network contention or bandwidth limits
   - Does not account for communication scheduling optimizations

2. **Synchronization Model**:
   - Assumes perfect barrier synchronization
   - Does not model asynchronous execution or pipelining

3. **Hardware Abstraction**:
   - Uses normalized time units (`COMPUTE_TIME_PER_TOKEN = 1.0`)
   - Communication costs are relative (`INTRANODE_COMM = 1`, `INTERNODE_COMM = 18`)
   - Actual hardware performance may vary

### Validation

The simulation model is based on:
- Standard MoE inference patterns in data-parallel settings
- Common communication patterns in distributed deep learning
- Established load balancing metrics (e.g., max/avg ratio)

### Comparison with Other Metrics

The script also calculates:
- **Compute-only balancedness**: `avg_compute_time / max_compute_time`
  - Measures only computation load balancing
  - Does not account for communication-induced idle time

- **Efficiency**: `avg_compute_time / max_elapsed_time`
  - Measures what fraction of total time is spent on computation
  - Lower values indicate more time spent on communication/synchronization

## Implementation Details

### Key Functions

1. **`simulate_elapsed_times(log2phy, logcnt, workload)`**
   - Core simulation function
   - Returns overall balancedness score and detailed metrics

2. **`evaluate_overall_balancedness(program_path, verbose)`**
   - Main evaluation function
   - Loads program, evaluates on all workload pairs
   - Returns aggregated results

### Constants

The script uses constants from `TAPE_evaluator.py`:
- `NUM_REPLICAS = 288`: Total number of physical experts
- `NUM_GPUS = 32`: Number of GPUs
- `NUM_NODES = 4`: Number of nodes
- `INTRANODE_COMM = 1`: Relative cost of intra-node communication
- `INTERNODE_COMM = 18`: Relative cost of inter-node communication

### Workload Processing

- Loads workloads from `data/expert-load.json`
- Processes workload pairs: uses `workload[i]` for rebalancing, evaluates on `workload[i+1]`
- Matches the evaluation protocol in `TAPE_evaluator.py`

## Integration with TAPE Evaluator

The overall balancedness metric complements the existing metrics in `TAPE_evaluator.py`:

- **`balancedness_score`**: Computation load balancing (avg/max load ratio)
- **`communication_score`**: Communication cost efficiency
- **`overall_balancedness`**: Combined elapsed time balancing (NEW)

Together, these metrics provide a comprehensive view of load balancing quality:
- Computation balancing (existing)
- Communication efficiency (existing)
- Overall elapsed time balancing (new)

## Future Improvements

Potential enhancements to the simulation model:

1. **Network Contention**: Model bandwidth limits and network contention
2. **Asynchronous Execution**: Support for pipelined/async execution models
3. **Hardware-Specific Timing**: Use actual hardware benchmarks for timing
4. **Communication Scheduling**: Model optimized communication schedules
5. **Memory Bandwidth**: Account for memory bandwidth limitations

## References

- Expert Parallelism Load Balancing (EPLB) for vLLM
- DeepSeek EPLB: https://github.com/deepseek-ai/eplb
- MoE (Mixture of Experts) architectures
- Data parallelism in distributed deep learning

