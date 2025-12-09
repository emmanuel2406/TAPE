#!/usr/bin/env python3
"""
Overall Balancedness Evaluator for Expert Parallelism Load Balancing

This script evaluates the overall balancedness of a rebalancing program by simulating
the actual execution timeline on GPUs, accounting for both computation time and idle
time due to communication dependencies.

Overall balancedness is defined as the balancing of elapsed time among GPUs, where
elapsed time = GPU compute time + idle time due to waiting on communication.

This metric combines computation balancing and communication-induced idle time balancing
into a single measure of how evenly GPUs are utilized over time.
"""

import argparse
import functools
import importlib.util
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Fallback if tqdm is not available
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import constants from TAPE_evaluator
EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
WORKLOAD_PATH = os.path.join(EVALUATOR_DIR, "data/expert-load.json")
REBALANCE_INTERVAL = 100

NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4

# Communication cost constants (matching TAPE_evaluator.py)
INTRANODE_COMM = 1
INTERNODE_COMM = 18
COMM_COMPUTATION_SCALE = 0.1

# Normalize computation time to 1.0 per token (can be adjusted based on actual hardware)
COMPUTE_TIME_PER_TOKEN = 1.0


@functools.cache
def load_workloads(path: str) -> List[torch.Tensor]:
    """Load workloads from the expert-load.json file."""
    with open(path, "r") as f:
        data = json.load(f)

    total_len = len(data['load_history'])
    workloads = []
    for i in range(0, total_len, REBALANCE_INTERVAL):
        start = i
        end = min(start + REBALANCE_INTERVAL, total_len)

        load = torch.tensor([x['logical_expert_load'] for x in data['load_history'][start:end]]).sum(dim=0)
        workloads.append(load)

    return workloads


def simulate_elapsed_times(
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    workload: torch.Tensor
) -> Tuple[float, Dict[str, float]]:
    """
    Simulate the execution timeline to calculate per-GPU elapsed times.
    
    This function models:
    1. Per-GPU compute time based on assigned load
    2. Communication time for routing tokens to expert replicas
    3. Idle time from waiting on communication dependencies
    4. Synchronization between layers
    
    Args:
        log2phy: [num_layers, num_logical_experts, max_replicas] - mapping from logical to physical experts
        logcnt: [num_layers, num_logical_experts] - number of replicas per logical expert
        workload: [num_layers, num_logical_experts] - tokens per logical expert per layer
    
    Returns:
        overall_balancedness: float - avg_elapsed_time / max_elapsed_time across all GPUs
        metrics: dict - detailed metrics including per-GPU times, compute times, idle times
    """
    num_layers, num_logical_experts = workload.shape
    num_physical_experts = NUM_REPLICAS
    phy_experts_per_gpu = num_physical_experts // NUM_GPUS
    gpus_per_node = NUM_GPUS // NUM_NODES
    
    # Initialize per-GPU time tracking
    # elapsed_time[gpu] = compute_time + idle_time
    gpu_compute_times = torch.zeros(NUM_GPUS, dtype=torch.float)
    gpu_idle_times = torch.zeros(NUM_GPUS, dtype=torch.float)
    gpu_elapsed_times = torch.zeros(NUM_GPUS, dtype=torch.float)
    
    # Track per-layer timing for analysis
    layer_times = []
    
    for layer_id in range(num_layers):
        # Step 1: Calculate per-GPU compute load for this layer
        gpu_compute_load = torch.zeros(NUM_GPUS, dtype=torch.float)
        
        for logical_id in range(num_logical_experts):
            logical_load = workload[layer_id][logical_id].item()
            if logical_load <= 0:
                continue
            
            num_replicas = int(logcnt[layer_id][logical_id].item())
            if num_replicas <= 0:
                continue
            
            # Get physical expert mapping
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
            
            # Distribute load evenly across replicas
            replica_load = logical_load / num_replicas
            
            # Assign load to GPUs based on physical expert placement
            for physical_id in physical_ids:
                target_gpu = int(physical_id.item()) // phy_experts_per_gpu
                gpu_compute_load[target_gpu] += replica_load
        
        # Step 2: Calculate per-GPU compute time
        # In data parallelism, each GPU processes its share of tokens
        # Compute time is proportional to the maximum load on that GPU
        layer_compute_times = gpu_compute_load * COMPUTE_TIME_PER_TOKEN
        
        # Step 3: Calculate communication time and dependencies
        # In MoE inference with data parallelism:
        # - Each GPU has (total_tokens / NUM_GPUS) tokens
        # - Tokens are routed to expert replicas on different GPUs
        # - Communication happens in parallel with computation when possible
        # - GPUs wait for all communication to complete before proceeding
        
        # Calculate communication time per GPU (both sending and receiving)
        # This represents the time to send/receive tokens for expert routing
        gpu_send_times = torch.zeros(NUM_GPUS, dtype=torch.float)
        gpu_recv_times = torch.zeros(NUM_GPUS, dtype=torch.float)
        
        total_tokens = workload[layer_id].sum().item()
        tokens_per_gpu = total_tokens / NUM_GPUS if NUM_GPUS > 0 else 0
        
        # Track per-target-GPU receive times (to take max across parallel receives)
        # recv_times_per_target[target_gpu] = list of receive times from different sources
        recv_times_per_target = {gpu: [] for gpu in range(NUM_GPUS)}
        
        for logical_id in range(num_logical_experts):
            num_tokens = workload[layer_id][logical_id].item()
            if num_tokens <= 0:
                continue
            
            num_replicas = int(logcnt[layer_id][logical_id].item())
            if num_replicas <= 0:
                continue
            
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
            tokens_per_gpu_per_replica = num_tokens / (NUM_GPUS * num_replicas)
            
            # Calculate communication time for each source GPU to send to target GPUs
            for source_gpu in range(NUM_GPUS):
                source_node = source_gpu // gpus_per_node
                max_send_time = 0.0
                
                for physical_id in physical_ids:
                    target_gpu = int(physical_id.item()) // phy_experts_per_gpu
                    target_node = target_gpu // gpus_per_node
                    
                    # No communication needed if same GPU
                    if source_gpu == target_gpu:
                        continue
                    
                    # Calculate communication cost
                    if source_node == target_node:
                        comm_cost = tokens_per_gpu_per_replica * INTRANODE_COMM
                    else:
                        comm_cost = tokens_per_gpu_per_replica * INTERNODE_COMM
                    
                    # Sends can happen in parallel, so take max
                    max_send_time = max(max_send_time, comm_cost)
                    
                    # Track receive time for target GPU (receives can happen in parallel)
                    recv_times_per_target[target_gpu].append(comm_cost)
                
                # Update send time (max across all sends from this GPU)
                gpu_send_times[source_gpu] = max(gpu_send_times[source_gpu], max_send_time)
        
        # Calculate receive times: takes max across parallel receives
        # In practice, receives from multiple sources can happen in parallel,
        # but bandwidth is shared. For a simplified model, we take the max
        # (conservative estimate - assumes perfect parallelization)
        for target_gpu in range(NUM_GPUS):
            if recv_times_per_target[target_gpu]:
                # Max receive time (parallel receives, take longest)
                gpu_recv_times[target_gpu] = max(recv_times_per_target[target_gpu])
        
        # Total communication time per GPU = max(send_time, recv_time)
        # Send and receive can overlap, so we take the maximum
        gpu_comm_times = torch.maximum(gpu_send_times, gpu_recv_times)
        
        # Step 4: Calculate per-GPU elapsed time for this layer
        # 
        # Model: Each GPU has work time = max(compute_time, comm_time)
        # Synchronization: all GPUs wait for the slowest, so layer completion = max(work_time)
        # 
        # For overall balancedness, we need to track how work is distributed.
        # The key insight: while synchronization forces all GPUs to wait for the slowest,
        # the work distribution (compute + comm) still matters because:
        # 1. GPUs with more work contribute more to the max (making others wait longer)
        # 2. Across multiple layers, GPUs with consistently more work will have
        #    higher total work times, even after accounting for synchronization
        #
        # We track individual GPU work times (before sync) to measure work distribution
        layer_work_times = torch.maximum(layer_compute_times, gpu_comm_times)
        
        # Layer completion time (synchronization barrier)
        max_layer_time = layer_work_times.max().item()
        
        # Calculate idle time per GPU: waiting for synchronization
        # idle_time = max_layer_time - GPU's own work_time
        layer_idle_times = torch.clamp(
            max_layer_time - layer_work_times,
            min=0.0
        )
        
        # Elapsed time per GPU for this layer (after synchronization)
        # All GPUs have the same elapsed time per layer due to synchronization
        layer_elapsed_times = torch.full_like(layer_work_times, max_layer_time)
        
        # Accumulate times across layers
        # Key insight: For overall balancedness, we measure work distribution
        # The elapsed time per layer is the same for all GPUs (due to sync),
        # but the WORK time (compute + comm) differs per GPU.
        # 
        # We track:
        # - Work time: individual GPU work (max(compute, comm)) - shows work distribution
        # - Elapsed time: work_time + idle_time - shows total time including sync wait
        #
        # For balancedness, we use work_time because it reflects how evenly
        # work is distributed. The sync overhead is captured separately in idle_time.
        gpu_compute_times += layer_compute_times
        gpu_idle_times += layer_idle_times
        # Accumulate individual GPU work times (not synchronized elapsed times)
        # This preserves the work distribution imbalance across layers
        gpu_elapsed_times += layer_work_times
        
        layer_times.append({
            'layer_id': layer_id,
            'max_compute_time': layer_compute_times.max().item(),
            'max_comm_time': gpu_comm_times.max().item(),
            'max_elapsed_time': max_layer_time,
            'avg_work_time': layer_work_times.mean().item(),
            'max_work_time': layer_work_times.max().item(),
        })
    
    # Step 6: Calculate overall balancedness
    # Overall balancedness = avg_elapsed_time / max_elapsed_time
    # This measures how evenly elapsed time is distributed across GPUs
    # 1.0 = perfect balance, < 1.0 = imbalanced
    
    avg_elapsed_time = gpu_elapsed_times.mean().item()
    max_elapsed_time = gpu_elapsed_times.max().item()
    min_elapsed_time = gpu_elapsed_times.min().item()
    
    if max_elapsed_time > 0:
        overall_balancedness = avg_elapsed_time / max_elapsed_time
    else:
        overall_balancedness = 1.0  # All GPUs idle, considered balanced
    
    # Calculate additional metrics
    avg_compute_time = gpu_compute_times.mean().item()
    max_compute_time = gpu_compute_times.max().item()
    avg_idle_time = gpu_idle_times.mean().item()
    max_idle_time = gpu_idle_times.max().item()
    
    # Compute-only balancedness (for comparison)
    compute_balancedness = avg_compute_time / max_compute_time if max_compute_time > 0 else 1.0
    
    metrics = {
        'overall_balancedness': overall_balancedness,
        'compute_balancedness': compute_balancedness,
        'avg_elapsed_time': avg_elapsed_time,
        'max_elapsed_time': max_elapsed_time,
        'min_elapsed_time': min_elapsed_time,
        'avg_compute_time': avg_compute_time,
        'max_compute_time': max_compute_time,
        'avg_idle_time': avg_idle_time,
        'max_idle_time': max_idle_time,
        'total_elapsed_time': max_elapsed_time,  # Total time = max across all GPUs
        'efficiency': avg_compute_time / max_elapsed_time if max_elapsed_time > 0 else 0.0,
        'layer_times': layer_times,
    }
    
    return overall_balancedness, metrics


def evaluate_overall_balancedness(program_path: str, verbose: bool = True) -> Dict:
    """
    Evaluate overall balancedness for a given rebalancing program.
    
    Args:
        program_path: Path to the best_program.py file
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary containing overall_balancedness score and detailed metrics
    """
    # Load workloads
    if not os.path.exists(WORKLOAD_PATH):
        raise FileNotFoundError(
            f"Workload file {WORKLOAD_PATH} not found. "
            "Please ensure the expert-load.json file exists."
        )
    
    workloads = load_workloads(WORKLOAD_PATH)
    
    # Load the program
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load program from {program_path}")
    
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    if not hasattr(program, "rebalance_experts"):
        raise ValueError(f"Program {program_path} does not have `rebalance_experts` function")
    
    # Evaluate on all workload pairs (same as TAPE_evaluator)
    overall_balancedness_scores = []
    all_metrics = []
    
    # Create progress bar
    workload_range = range(len(workloads) - 1)
    if verbose and TQDM_AVAILABLE:
        workload_range = tqdm(workload_range, desc="Evaluating workloads", unit="workload")
    
    for i in workload_range:
        # Get rebalancing from workload[i]
        _, log2phy, logcnt = program.rebalance_experts(
            workloads[i],
            NUM_REPLICAS,
            NUM_GROUPS,
            NUM_NODES,
            NUM_GPUS,
        )
        
        # Evaluate on workload[i+1]
        balancedness, metrics = simulate_elapsed_times(log2phy, logcnt, workloads[i + 1])
        overall_balancedness_scores.append(balancedness)
        all_metrics.append(metrics)
        
        # Update progress bar with current score if verbose and tqdm is available
        if verbose and TQDM_AVAILABLE and hasattr(workload_range, 'set_postfix'):
            avg_score = sum(overall_balancedness_scores) / len(overall_balancedness_scores)
            workload_range.set_postfix({
                'balancedness': f'{balancedness:.4f}',
                'avg': f'{avg_score:.4f}'
            })
    
    # Calculate average metrics
    avg_overall_balancedness = sum(overall_balancedness_scores) / len(overall_balancedness_scores)
    avg_compute_balancedness = sum(m['compute_balancedness'] for m in all_metrics) / len(all_metrics)
    avg_efficiency = sum(m['efficiency'] for m in all_metrics) / len(all_metrics)
    avg_total_time = sum(m['total_elapsed_time'] for m in all_metrics) / len(all_metrics)
    
    result = {
        'overall_balancedness': float(avg_overall_balancedness),
        'compute_balancedness': float(avg_compute_balancedness),
        'efficiency': float(avg_efficiency),
        'avg_total_elapsed_time': float(avg_total_time),
        'num_workloads': len(workloads) - 1,
        'per_workload_scores': [float(s) for s in overall_balancedness_scores],
    }
    
    if verbose:
        print(f"Overall Balancedness: {avg_overall_balancedness:.6f}")
        print(f"  (Compute-only balancedness: {avg_compute_balancedness:.6f})")
        print(f"  (Efficiency: {avg_efficiency:.6f})")
        print(f"  (Avg total elapsed time: {avg_total_time:.2f})")
        print(f"  (Evaluated on {len(workloads) - 1} workload pairs)")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate overall balancedness of an expert parallelism load balancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s openevolve/examples/eplb/output/best/best_program.py
  %(prog)s --quiet openevolve/examples/eplb/output/best/best_program.py
        """
    )
    parser.add_argument(
        'program_path',
        type=str,
        help='Path to the best_program.py file to evaluate'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.program_path):
        print(f"Error: Program file not found: {args.program_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        result = evaluate_overall_balancedness(args.program_path, verbose=not args.quiet)
        
        if args.json:
            import json
            print(json.dumps(result, indent=2))
        else:
            if not args.quiet:
                print("\nDetailed per-workload scores:")
                for i, score in enumerate(result['per_workload_scores']):
                    print(f"  Workload {i+1}: {score:.6f}")
        
        # Exit with code 0 on success
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

