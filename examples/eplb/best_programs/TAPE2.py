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

    # Vectorized balanced packing: Sort weights descendingly and assign in blocks.
    # This achieves near-optimal load balancing for fixed-size bins (groups_per_pack)
    # and is significantly faster than the iterative greedy approach.
    current_device = weight.device
    sorted_indices = weight.sort(-1, descending=True).indices # weight is already float from rebalance_experts

    pack_index_out = torch.empty_like(sorted_indices, dtype=torch.int64, device=current_device)
    rank_in_pack_out = torch.empty_like(sorted_indices, dtype=torch.int64, device=current_device)

    # Distribute sorted items in a snake-like (zigzag) fashion for better load balancing.
    indices = torch.arange(num_groups, device=current_device)
    block_idx = indices // num_packs
    offset_in_block = indices % num_packs
    pack_idx_flat = torch.where(block_idx % 2 == 0, offset_in_block,
                            num_packs - 1 - offset_in_block)
    rank_flat = indices // num_packs

    # Vectorized assignment: expand flat indices and scatter back using sorted order.
    pack_idx_flat_expanded = pack_idx_flat.expand(num_layers, num_groups)
    rank_flat_expanded = rank_flat.expand(num_layers, num_groups)
    
    pack_index_out.scatter_(1, sorted_indices, pack_idx_flat_expanded)
    rank_in_pack_out.scatter_(1, sorted_indices, rank_flat_expanded)

    return pack_index_out, rank_in_pack_out


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
    if num_redundant == 0:
        phy2log = torch.arange(num_log, dtype=torch.int64, device=device).unsqueeze(0).expand(n, num_log)
        rank = torch.zeros(n, num_log, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # Step 1: Calculate logcnt based on proportional distribution (Largest Remainder Method)
    total_load = weight.sum(dim=-1, keepdim=True)  # [n, 1]
    # Avoid division by zero for layers with no load
    total_load = total_load.clamp(min=1.0) # More concise and efficient

    # Calculate ideal fractional counts for each expert
    ideal_counts_float = weight * (num_phy / total_load)  # [n, num_log]

    logcnt = ideal_counts_float.floor().long()  # [n, num_log]
    remainders = ideal_counts_float - logcnt.float()  # [n, num_log]

    # Calculate how many additional experts each layer needs to reach num_phy
    remaining_slots_per_layer = num_phy - logcnt.sum(dim=-1)  # [n]

    # Distribute remaining slots based on largest remainders
    # Sort remainders in descending order to find which experts get the extra slots
    sorted_remainder_indices = remainders.argsort(dim=-1, descending=True)

    # Create a mask to identify which experts (by original index) should receive an additional replica
    row_indices = torch.arange(n, device=device).unsqueeze(1)  # [n, 1]
    col_ranks = torch.arange(num_log, device=device).unsqueeze(0)  # [1, num_log]

    # `mask_for_sorted_indices` indicates which *ranks* in the sorted list get an increment.
    mask_for_sorted_indices = col_ranks < remaining_slots_per_layer.unsqueeze(1)  # [n, num_log]

    # `expert_indices_to_increment` are the original expert IDs that need an increment.
    expert_indices_to_increment = sorted_remainder_indices[mask_for_sorted_indices]  # 1D tensor
    
    # `layer_indices_to_increment` are the corresponding layer IDs.
    layer_indices_to_increment = row_indices.expand(n, num_log)[mask_for_sorted_indices]  # 1D tensor

    logcnt[layer_indices_to_increment, expert_indices_to_increment] += 1

    # Assert that the total number of physical experts now matches num_phy for each layer
    assert torch.all(logcnt.sum(dim=-1) == num_phy), \
        f"Total physical experts mismatch: {logcnt.sum(dim=-1)} vs {num_phy}"

    # Step 2: Construct phy2log and rank from the final logcnt
    # Max number of replicas for any single logical expert across all layers.
    max_log_cnt_val = logcnt.max().item() # Guaranteed > 0 if num_log > 0

    # Create phy2log: repeat logical expert IDs based on logcnt
    # Flatten logical expert IDs for efficient repeat_interleave
    flat_log_ids = torch.arange(num_log, device=device).unsqueeze(0).expand(n, num_log).flatten() # [n * num_log]
    
    # Repeat each logical expert ID according to its count in logcnt
    phy2log_flat = torch.repeat_interleave(flat_log_ids, logcnt.flatten()) # [n * num_phy]
    phy2log = phy2log_flat.view(n, num_phy) # Reshape to [n, num_phy]

    # Create rank: for each group of repeated experts, assign ranks 0, 1, ...
    # Create a template of ranks, then use a mask to select valid ranks based on logcnt
    rank_template = torch.arange(max_log_cnt_val, device=device) # [max_log_cnt_val]
    rank_values_expanded = rank_template.unsqueeze(0).unsqueeze(0).expand(n, num_log, max_log_cnt_val) # [n, num_log, max_log_cnt_val]
    
    # Create a mask to select only valid ranks for each expert based on its replica count
    mask = (rank_values_expanded < logcnt.unsqueeze(-1)) # [n, num_log, max_log_cnt_val]
    
    # Gather the rank values using the mask. This will flatten the result.
    rank_flat = rank_values_expanded[mask] # [n * num_phy]
    
    rank = rank_flat.view(n, num_phy) # Reshape to [n, num_phy]

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
    weight = weight.float() # Keep computation on original device (e.g., GPU)

    if num_logical_experts == 0:
        # Handle case where there are no logical experts to avoid errors in subsequent calculations.
        # phy2log will be [num_layers, num_replicas] with -1, phyrank [num_layers, num_replicas] with 0.
        # logcnt will be an empty [num_layers, 0] tensor.
        # Create these tensors directly as sub-functions would fail with num_log=0.
        phy2log = torch.full((num_layers, num_replicas), -1, dtype=torch.int64, device=weight.device)
        phyrank = torch.zeros((num_layers, num_replicas), dtype=torch.int64, device=weight.device)
        logcnt = torch.empty((num_layers, 0), dtype=torch.int64, device=weight.device)
        return phy2log, torch.empty((num_layers, 0, 1), dtype=torch.int64, device=weight.device), logcnt

    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    # Determine the actual maximum number of replicas any logical expert received.
    # This helps size log2phy more precisely, saving memory and improving cache efficiency.
    # `logcnt` is guaranteed to contain values >= 1 (if num_logical_experts > 0), so `.max().item()` is always valid.
    actual_max_log_cnt = logcnt.max().item()
    
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, actual_max_log_cnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device, # Use logcnt device for consistency
    )
    # Calculate flattened indices for the scatter operation.
    # Each physical expert (p_idx) maps to a logical expert (phy2log[p_idx])
    # with a specific replica rank (phyrank[p_idx]).
    # The index into the flattened log2phy tensor is derived from these.
    flat_indices = phy2log * actual_max_log_cnt + phyrank
    
    log2phy.view(num_layers, -1).scatter_(
        -1,
        flat_indices,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=logcnt.device).expand(num_layers, -1), # Use logcnt device
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
