import functools
import importlib.util
import json
import os
import time
import traceback
from typing import TypedDict

import torch

# Get the directory where this evaluator file is located
EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
WORKLOAD_PATH = os.path.join(EVALUATOR_DIR, "data/expert-load.json")
REBALANCE_INTERVAL = 100

NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4


# Ratio of GPU-to-GPU communication cost within a node vs GPU-to-GPU communication cost between nodes
INTRANODE_COMM = 1
INTERNODE_COMM = 18
COMM_SCORE_SCALE = 10
COMM_COMPUTATION_SCALE = 0.1

# Weight copying cost constants
# Number of bytes for 1 expert's parameters (e.g., for a typical MoE expert with ~1B parameters, this would be ~4GB for float32)
BYTES_PER_EXPERT = 4_000_000_000  # 4GB per expert (adjust based on your model)COMM
WEIGHT_COPY_SCORE_SCALE = 0.5

# Check if workload file exists
if not os.path.exists(WORKLOAD_PATH):
    raise FileNotFoundError(f"Workload file {WORKLOAD_PATH} not found. "
        "Please download the workload file as instructed in the `README.md` "
        "under the `eplb` directory."
    )

@functools.cache
def load_workloads(path: str) -> list[torch.Tensor]: 
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

class EvaluationResult(TypedDict, total=False):
    balancedness_score: float
    speed_score: float
    communication_score: float
    weight_copy_score: float
    combined_score: float
    error: str

def calculate_weight_copy_cost(log2phy: torch.Tensor, logcnt: torch.Tensor) -> float:
    '''
    Calculate the communication cost for copying replica weights from source GPUs to replica GPUs.
    Returns a normalized efficiency score (higher is better, similar to communication_efficiency).
    '''
    num_layers, num_logical_experts, _ = log2phy.shape
    
    # Calculate mapping from physical experts to GPUs and nodes
    num_physical_experts = NUM_REPLICAS
    phy_experts_per_gpu = num_physical_experts // NUM_GPUS
    gpus_per_node = NUM_GPUS // NUM_NODES
    
    total_weight_copy_cost = 0.0
    min_weight_copy_cost = 0.0  # Theoretical minimum (all intra-node copies)
    
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            num_replicas = int(logcnt[layer_id][logical_id].item())
            
            # Skip if no replicas (no copying needed)
            if num_replicas <= 1:
                continue
            
            # Get physical expert mapping
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
            
            # The first physical expert is the source (original)
            source_physical_id = int(physical_ids[0].item())
            source_gpu = source_physical_id // phy_experts_per_gpu
            source_node = source_gpu // gpus_per_node
            
            # The remaining physical experts are replicas that need weight copying
            replica_physical_ids = physical_ids[1:]
            
            # Calculate minimum cost: all replicas on same node as source
            # In optimal case, we place replicas on same node (intra-node cost)
            num_replicas_to_copy = len(replica_physical_ids)
            replicas_on_same_node = min(num_replicas_to_copy, gpus_per_node - 1)  # -1 because source GPU already has the expert
            min_weight_copy_cost += replicas_on_same_node * BYTES_PER_EXPERT * INTRANODE_COMM
            
            # Calculate actual cost
            for replica_physical_id in replica_physical_ids:
                target_physical_id = int(replica_physical_id.item())
                target_gpu = target_physical_id // phy_experts_per_gpu
                target_node = target_gpu // gpus_per_node
                
                # If source and target are on same GPU, no copying needed (shouldn't happen for replicas, but check anyway)
                if source_gpu == target_gpu:
                    continue
                
                # Determine communication cost based on node placement
                if source_node == target_node:
                    # Intra-node communication
                    copy_cost = BYTES_PER_EXPERT * INTRANODE_COMM
                else:
                    # Inter-node communication
                    copy_cost = BYTES_PER_EXPERT * INTERNODE_COMM
                
                total_weight_copy_cost += copy_cost
    
    # Normalize weight copy cost (min / actual)
    # This gives a ratio where 1.0 is optimal (minimum cost) and < 1.0 is worse
    weight_copy_efficiency = WEIGHT_COPY_SCORE_SCALE * min_weight_copy_cost / total_weight_copy_cost if total_weight_copy_cost > 0 else 0.0
    
    print(f'weight_copy_cost: {total_weight_copy_cost}, min_weight_copy_cost: {min_weight_copy_cost}, weight_copy_efficiency: {weight_copy_efficiency}')
    
    return weight_copy_efficiency

def simulate_inference(log2phy: torch.Tensor, logcnt: torch.Tensor, workload: torch.Tensor) -> tuple[float, float]:
    '''
    Simulate a MoE inference with the given expert mapping, and return the balancedness factor and the communication cost.
    '''
    # workload shape: (num_layers, num_logical_experts) - load per logical expert per layer
    num_layers, num_logical_experts = workload.shape
    
    # Initialize physical expert load accumulator
    num_physical_experts = NUM_REPLICAS
    total_physical_load = torch.zeros(num_layers, num_physical_experts, dtype=torch.float, device=workload.device)
    
    # Calculate mapping from physical experts to GPUs and nodes
    phy_experts_per_gpu = num_physical_experts // NUM_GPUS
    gpus_per_node = NUM_GPUS // NUM_NODES
    
    # For each logical expert, distribute load to its physical replicas
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            # Get the load for this logical expert
            logical_load = workload[layer_id][logical_id].item()
            
            # Skip zero load
            if logical_load <= 0:
                continue
                
            num_replicas = int(logcnt[layer_id][logical_id].item())

            # Skip zero replicas
            if num_replicas <= 0:
                continue

            # Get physical expert mapping
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
                
            # Calculate load per replica (based on number of valid replicas)
            replica_load = logical_load / num_replicas
            
            # Assign load to valid physical experts
            total_physical_load[layer_id, physical_ids] += replica_load
    
    # Calculate balancedness
    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0, 0.0
    
    # Calculate average and max load per layer, then sum
    layer_avg = total_physical_load.mean(dim=1)  # (num_layers,)
    layer_max = total_physical_load.max(dim=1).values  # (num_layers,)
    
    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()
    
    # Calculate balancedness: avg_load / max_load
    balancedness = avg_load / max_load if max_load > 0 else 0.0
    
    # Calculate communication cost
    # In data parallelism, tokens are uniformly distributed across all GPUs
    # Each GPU processes (num_tokens / num_gpus) tokens
    # When tokens need to be routed to experts, they must be sent from source GPU to target GPU (where the expert is located)
    # For each logical expert, tokens are distributed to its physical replicas
    # Each replica processes (num_tokens / num_replicas) tokens
    # Each GPU sends its share of tokens to each replica
    total_communication_cost = 0.0
    min_communication_cost = 0.0  # Theoretical minimum (all intra-node, optimally distributed)
    
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            # Get the number of tokens for this logical expert
            num_tokens = workload[layer_id][logical_id].item()
            
            # Skip zero tokens
            if num_tokens <= 0:
                continue
            
            num_replicas = int(logcnt[layer_id][logical_id].item())
            if num_replicas <= 0:
                continue
            
            # Get physical expert mapping
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
            
            # In data parallelism, tokens are uniformly distributed across all GPUs
            # Each GPU has (num_tokens / NUM_GPUS) tokens
            # These tokens need to be sent to expert replicas
            # Each replica processes (num_tokens / num_replicas) tokens
            # Each GPU sends (num_tokens / NUM_GPUS) / num_replicas tokens to each replica
            tokens_per_gpu_per_replica = num_tokens / (NUM_GPUS * num_replicas)
            
            # Calculate minimum communication cost for this logical expert
            # Minimum occurs when all replicas are placed to minimize communication
            # In data parallelism, tokens are uniformly distributed across all GPUs
            # Optimal strategy: place replicas distributed across all nodes to minimize inter-node comm
            # For each source GPU, optimal placement ensures:
            #   - One replica on same GPU (0 cost) if possible
            #   - Remaining replicas on same node (intra-node cost)
            #   - No inter-node communication
            # This is achievable by distributing replicas across all nodes
            min_comm_for_expert = 0.0
            for source_gpu in range(NUM_GPUS):
                source_node = source_gpu // gpus_per_node
                # In optimal case: distribute replicas across GPUs on same node as source
                # At most one replica per GPU, so we can place min(num_replicas, gpus_per_node) on same node
                # One of these can be on same GPU (0 cost), rest on same node (intra-node)
                replicas_on_same_node = min(num_replicas, gpus_per_node)
                replicas_on_same_gpu = min(1, replicas_on_same_node)  # At most one on same GPU
                replicas_requiring_intranode = replicas_on_same_node - replicas_on_same_gpu
                # Same GPU replica has 0 cost (skipped)
                # Same node replicas (excluding same GPU) have intra-node cost
                min_comm_for_expert += tokens_per_gpu_per_replica * replicas_requiring_intranode * INTRANODE_COMM
                # If we have more replicas than GPUs on same node, they must go to other nodes
                # But for minimum, we assume optimal distribution keeps them on same node
                # (In practice, if num_replicas > gpus_per_node, some must go elsewhere)
            
            min_communication_cost += min_comm_for_expert
            
            for physical_id in physical_ids:
                # Determine the GPU and node where the physical expert is located
                target_gpu = int(physical_id.item()) // phy_experts_per_gpu
                target_node = target_gpu // gpus_per_node
                
                # For each source GPU, calculate communication cost to target GPU
                for source_gpu in range(NUM_GPUS):
                    source_node = source_gpu // gpus_per_node
                    
                    # If source GPU and target GPU are the same, no communication needed
                    if source_gpu == target_gpu:
                        continue
                    
                    # Determine communication cost (scaled by number of tokens)
                    if source_node == target_node:
                        # Intra-node communication
                        comm_cost = tokens_per_gpu_per_replica * INTRANODE_COMM
                    else:
                        # Inter-node communication
                        comm_cost = tokens_per_gpu_per_replica * INTERNODE_COMM
                    
                    total_communication_cost += comm_cost
    
    # Calculate total computation cost (sum of all tokens processed)
    total_computation_cost = workload.sum().item()
    
    # Calculate exposed communication: only the communication cost beyond what's hidden by computation
    # Use ReLU(communication - Z * computation) where Z is COMM_COMPUTATION_SCALE
    exposed_communication_cost = torch.clamp(
        torch.tensor(total_communication_cost) - COMM_COMPUTATION_SCALE * total_computation_cost,
        min=0.0
    ).item()
    exposed_min_communication_cost = torch.clamp(
        torch.tensor(min_communication_cost) - COMM_COMPUTATION_SCALE * total_computation_cost,
        min=0.0
    ).item()
    print("COMMUNICATION COST: ", total_computation_cost, min_communication_cost)
    
    # Normalize communication cost similar to balancedness (min / actual)
    # This gives a ratio where 1.0 is optimal (minimum cost) and < 1.0 is worse
    # Similar to balancedness where 1.0 is perfect balance and < 1.0 is imbalanced
    # Use exposed communication costs instead of raw communication costs
    communication_efficiency = COMM_SCORE_SCALE * exposed_min_communication_cost / exposed_communication_cost if exposed_communication_cost > 0 else 0.0
    
    print(f'balancedness: {balancedness} avg_load: {avg_load}, max_load: {max_load}, communication_cost: {total_communication_cost}, min_comm_cost: {min_communication_cost}, computation_cost: {total_computation_cost}, exposed_comm_cost: {exposed_communication_cost}, exposed_min_comm_cost: {exposed_min_communication_cost}, communication_efficiency: {communication_efficiency}')
    
    return balancedness, communication_efficiency

def evaluate(program_path: str) -> EvaluationResult:
    workloads = load_workloads(WORKLOAD_PATH)

    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        assert spec is not None
        program = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(program)

        if not hasattr(program, "rebalance_experts"):
            print('Error: program does not have `rebalance_experts` function')
            return {
                "balancedness_score": 0.0,
                "speed_score": 0.0,
                "communication_score": 0.0,
                "weight_copy_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing `rebalance_experts` function",
            }

        if not hasattr(program, "rebalance_experts"):
            raise ValueError("Program does not have rebalance_experts function")
        
        balancedness_scores = []
        communication_scores = []
        weight_copy_scores = []
        times = []
        raw_times = []
        for i in range(len(workloads) - 1):
            start_time = time.perf_counter()
            _, log2phy, logcnt = program.rebalance_experts(
                workloads[i],
                NUM_REPLICAS,
                NUM_GROUPS,
                NUM_NODES,
                NUM_GPUS,
            )
            end_raw_time = time.perf_counter()
            balancedness_score, communication_score = simulate_inference(log2phy, logcnt, workloads[i + 1])
            weight_copy_score = calculate_weight_copy_cost(log2phy, logcnt)
            end_time = time.perf_counter()
            balancedness_scores.append(balancedness_score)
            communication_scores.append(communication_score)
            weight_copy_scores.append(weight_copy_score)
            times.append(end_time - start_time)
            raw_times.append(end_raw_time - start_time)
        avg_balancedness_score = sum(balancedness_scores) / len(balancedness_scores)
        avg_communication_score = sum(communication_scores) / len(communication_scores)
        avg_weight_copy_score = sum(weight_copy_scores) / len(weight_copy_scores)
        avg_time = sum(times) / len(times)
        avg_raw_time = sum(raw_times) / len(raw_times)
        speed_score = 0.02 / avg_time
        print(f'avg_time: {avg_time}, avg_raw_time: {avg_raw_time}, speed_score: {speed_score}')
        combined_score = (avg_balancedness_score + speed_score + avg_communication_score + avg_weight_copy_score) / 4
        return {
            "balancedness_score": float(avg_balancedness_score),
            "speed_score": float(speed_score),
            "avg_raw_time": float(avg_raw_time),
            "communication_score": float(avg_communication_score),
            "weight_copy_score": float(avg_weight_copy_score),
            "combined_score": float(combined_score),
        }
    except Exception as e:
        traceback.print_exc()
        print(f'Error during evaluation: {str(e)}')
        return {
            "balancedness_score": 0.0,
            "speed_score": 0.0,
            "communication_score": 0.0,
            "weight_copy_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }
    
    return {
        "balancedness_score": 0.0,
        "speed_score": 0.0,
        "communication_score": 0.0,
        "weight_copy_score": 0.0,
        "combined_score": 0.0,
        "error": "No error",
    }
    