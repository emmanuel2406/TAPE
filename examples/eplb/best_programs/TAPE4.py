# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

# EVOLVE-BLOCK-START

import torch
import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.
    This version is vectorized across `num_layers` for improved efficiency.

    Parameters:
        weight: [X, n], the weight of each item (X is num_layers)
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        # This case is already fully vectorized and efficient
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # `weight` is already float and on CPU from `rebalance_experts`
    device = weight.device

    # Sort items by weight in descending order for all layers simultaneously
    # original_indices stores the original column index for each sorted item
    sorted_weights, original_indices = torch.sort(weight, dim=-1, descending=True)
    
    # Initialize output tensors
    pack_index = torch.full_like(weight, -1, dtype=torch.int64)
    rank_in_pack = torch.full_like(weight, -1, dtype=torch.int64)

    # Initialize pack states for all layers, vectorized
    pack_weights = torch.zeros(num_layers, num_packs, dtype=torch.float32, device=device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)

    # Create a tensor for layer indices to enable advanced indexing
    layer_indices_range = torch.arange(num_layers, device=device)

    # Iterate through each item position (from heaviest to lightest) for all layers
    # This loop runs `num_groups` times, with vectorized operations inside.
    # It replaces the two nested Python loops in the original implementation.
    for k in range(num_groups): # k-th heaviest item for each layer
        # Get the original index and weight of the current k-th heaviest item for each layer
        current_original_group_idx = original_indices[:, k] # Shape: [num_layers]
        current_item_weights = sorted_weights[:, k]         # Shape: [num_layers]

        # Find the minimum weight pack for each layer, respecting pack capacity.
        # Use torch.where to mask full packs with a very large value for argmin.
        masked_pack_weights = torch.where(pack_items < groups_per_pack, 
                                          pack_weights, 
                                          torch.finfo(torch.float32).max)

        # Find the index of the minimum weight pack for each layer
        packs_to_assign = torch.argmin(masked_pack_weights, dim=-1) # Shape: [num_layers]

        # Assign the current item to its chosen pack for each layer
        # Using advanced indexing (layer_indices_range, current_original_group_idx)
        pack_index[layer_indices_range, current_original_group_idx] = packs_to_assign
        rank_in_pack[layer_indices_range, current_original_group_idx] = \
            pack_items[layer_indices_range, packs_to_assign]
        
        # Update pack weights and item counts for the assigned packs
        pack_weights[layer_indices_range, packs_to_assign] += current_item_weights
        pack_items[layer_indices_range, packs_to_assign] += 1
            
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    assert num_phy >= num_log  # Ensure we have enough physical experts for all logical ones
    device = weight.device

    if num_log == 0:
        # No logical experts to replicate. All physical experts are "empty".
        # This can happen if num_logical_experts < num_nodes in hierarchical balancing.
        phy2log = torch.full((n, num_phy), -1, dtype=torch.int64, device=device)
        rank = torch.full((n, num_phy), -1, dtype=torch.int64, device=device)
        logcnt = torch.zeros(n, 0, dtype=torch.int64, device=device) # Empty tensor
        return phy2log, rank, logcnt

    # Initialize phy2log and rank with -1, and logcnt with 0,
    # to allow all physical experts to be assigned greedily from scratch.
    phy2log = torch.full((n, num_phy), -1, dtype=torch.int64, device=device)
    rank = torch.full((n, num_phy), -1, dtype=torch.int64, device=device)
    logcnt = torch.zeros(n, num_log, dtype=torch.int64, device=device)  # Start with 0 replicas for all logical experts
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Use a small epsilon to avoid division by zero when logcnt is 0
    # and to ensure experts with 0 replicas are correctly prioritized (effectively infinite load per replica).
    epsilon = 1e-6 

    # Iterate for ALL physical experts (from 0 to num_phy-1)
    # This ensures all physical experts are assigned greedily based on current load.
    for i in range(num_phy): 
        # Calculate load per replica. Add epsilon to logcnt to handle initial 0.
        current_load_per_replica = weight / (logcnt.float() + epsilon)
        
        # Find the logical expert with the maximum load per replica for each layer
        # This is a greedy choice to balance load.
        redundant_indices = current_load_per_replica.max(dim=-1).indices
        
        # Assign the current physical expert (i) to the chosen logical expert
        phy2log[arangen, i] = redundant_indices
        
        # The rank is the number of replicas this logical expert already has *before* this assignment
        rank[arangen, i] = logcnt[arangen, redundant_indices]
        
        # Increment the replica count for the chosen logical expert
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: Pack physical_experts to GPUs, prioritizing co-locating replicas.
    # This aims to reduce weight copy cost (Goal 4) by grouping replicas of the same logical expert
    # onto the same GPU, while still distributing experts for load balance through sequential assignment.
    P_prime_phy = num_physical_experts // num_nodes
    
    # Calculate shift_bits dynamically to create a unique combined key for sorting.
    # The maximum replica rank for any logical expert cannot exceed P_prime_phy.
    # P_prime_phy.bit_length() gives the number of bits required to represent max rank.
    shift_bits = P_prime_phy.bit_length()
    if shift_bits == 0: shift_bits = 1 # Ensure at least 1 bit for small P_prime_phy values.

    # Create a combined key for sorting: (logical_expert_id << shift_bits) | replica_rank.
    # This ensures that physical experts mapping to the same logical expert are grouped together,
    # and within each logical expert, they are ordered by their replica rank.
    # phy2mlog and phyrank are [num_layers * num_nodes, P_prime_phy]
    combined_key = (phy2mlog.long() << shift_bits) | phyrank.long()

    # Sort physical experts based on this combined key.
    # `pphy2phy` will be the permutation that reorders physical expert indices
    # such that replicas of the same logical expert are contiguous.
    _, pphy2phy = torch.sort(combined_key, dim=-1)

    # Calculate the inverse permutation for subsequent gather operations.
    # This variable is not used in the current implementation, but kept for consistency with original.
    phy2pphy = inverse(pphy2phy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, P_prime_phy]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    # Calculate the maximum actual number of replicas any logical expert received.
    # This is a tighter bound for maxlogcnt, potentially reducing memory and improving efficiency.
    # logcnt is [num_layers, num_logical_experts]
    if num_logical_experts > 0:
        actual_max_replica_count = logcnt.max().item()
    else:
        # If there are no logical experts, then no replicas are assigned.
        # Set actual_max_replica_count to 0.
        actual_max_replica_count = 0
    
    # maxlogcnt must be large enough to hold the maximum possible replica rank (0-indexed).
    # It should be at least `actual_max_replica_count`.
    # Ensure it's at least 1 if `num_logical_experts > 0` to avoid potential issues with
    # zero-sized dimensions in some PyTorch operations, though `(N, M, 0)` is generally valid.
    # If num_logical_experts is 0, maxlogcnt can be 0.
    maxlogcnt = max(1, actual_max_replica_count) if num_logical_experts > 0 else 0
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
