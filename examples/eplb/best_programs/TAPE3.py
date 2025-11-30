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


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    device = weight.device
    
    # Sort indices on the original device. weight is already float from rebalance_experts.
    indices = weight.sort(-1, descending=True).indices # [num_layers, num_groups]

    # Initialize output tensors on the original device.
    pack_index = torch.full((num_layers, num_groups),
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device=device)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    
    # Use tensors for pack_weights and pack_items, now tracking state for all layers simultaneously.
    # These tensors will accumulate weights and item counts across all layers.
    pack_weights = torch.zeros(num_layers, num_packs, dtype=torch.float, device=device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)

    # Pre-calculate arange for advanced indexing, to be used repeatedly inside the loop.
    layer_indices_arange = torch.arange(num_layers, device=device)

    # Pre-allocate infinity tensor once to avoid repeated allocation in the loop.
    inf_tensor = torch.full((num_layers, num_packs), float('inf'), dtype=torch.float, device=device)

    # Iterate over sorted group positions, processing all layers in parallel for each position.
    # This replaces the outer 'for i in range(num_layers)' Python loop.
    for k in range(num_groups):
        # Get the k-th heaviest expert's original index for each layer.
        current_group_indices = indices[:, k] # Shape: [num_layers]
        # Get the corresponding weights for these experts across all layers.
        current_group_weights = weight[layer_indices_arange, current_group_indices] # Shape: [num_layers]

        # Determine eligible packs for each layer (those not yet full).
        eligible_mask = (pack_items < groups_per_pack) # Shape: [num_layers, num_packs]
        
        # Temporarily set non-eligible pack weights to infinity to exclude them from argmin.
        temp_pack_weights = torch.where(eligible_mask, pack_weights, inf_tensor) # Shape: [num_layers, num_packs]
        
        # Find the index of the least loaded eligible pack for each layer.
        packs_to_assign = temp_pack_weights.argmin(dim=1) # Shape: [num_layers]

        # Assign the group to the chosen pack for each layer.
        pack_index[layer_indices_arange, current_group_indices] = packs_to_assign
        
        # Get the current item count in the chosen pack for each layer, to use as rank.
        current_pack_items = pack_items[layer_indices_arange, packs_to_assign] # Shape: [num_layers]
        rank_in_pack[layer_indices_arange, current_group_indices] = current_pack_items
            
        # Update pack weights and item counts for the chosen packs across all layers.
        pack_weights[layer_indices_arange, packs_to_assign] += current_group_weights
        pack_items[layer_indices_arange, packs_to_assign] += 1
            
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
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    # Initialize phy2log for the base num_log experts (1:1 mapping, rank 0)
    # The remaining num_redundant slots in phy2log and rank will be filled in the loop.
    phy2log = torch.empty((n, num_phy), dtype=torch.int64, device=device)
    phy2log[:, :num_log] = torch.arange(num_log, dtype=torch.int64, device=device).expand(n, num_log)
    
    rank = torch.empty((n, num_phy), dtype=torch.int64, device=device)
    rank[:, :num_log] = 0 # All base experts get rank 0

    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    
    # Distribute redundant experts greedily
    for i in range(num_log, num_phy):
        # Use composite score for tie-breaking: prioritize experts
        # with higher absolute weight when the load ratio (weight/logcnt) is tied.
        # This is a robust LPT variant for replication scheduling.
        composite_score = (weight / logcnt) * 1e9 + weight
        redundant_indices = composite_score.max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
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

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
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
    # Ensure weight is float and capture its device.
    # The load balancer should operate on the device where the weights reside.
    weight = weight.float()
    device = weight.device # Capture the original device for consistent device placement

    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
            
    # Calculate the actual maximum number of replicas any single logical expert received
    # across all layers. This is a more precise bound for the log2phy tensor,
    # reducing memory usage and potentially speeding up the scatter_ operation.
    actual_max_log_count = logcnt.max().item()

    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, actual_max_log_count), # Use actual max count
        -1,
        dtype=torch.int64,
        device=device, # Use the original device for consistency
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * actual_max_log_count + phyrank, # Adjust linear index calculation
        torch.arange(num_replicas, dtype=torch.int64,
                     device=device).expand(num_layers, -1), # Use original device
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
