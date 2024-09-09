# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x

def get_current_cls_token_pos_from_source(original_csl_token_positions,source): #Todo -returns value, var name suggests index
    if(source== None):
        return original_csl_token_positions
    new_pos_of_orig_token = source.argmax(2)
    device = source.device
    batch_idx = torch.arange(original_csl_token_positions.shape[0]).unsqueeze(1).to(device)

    new_cls_token_pos = new_pos_of_orig_token[batch_idx,original_csl_token_positions.unsqueeze(1)].squeeze(1) #should never get merged-> only one orig token corresponds to new cls token
    return new_cls_token_pos

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    source,
    original_csl_token_positions #original_csl_token_positions
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    def set_cls_token_scores_to_mininf(scores):
        device = scores.device
        cls_pos = get_current_cls_token_pos_from_source(original_csl_token_positions,source) # dim: b, 1
        cls_is_in_a = cls_pos%2 == 0 #mask, in which batch elem the cls token is in a
        batch_idx = torch.arange(original_csl_token_positions.shape[0]).unsqueeze(1).to(device)
        cls_in_a_batch_idx = batch_idx[cls_is_in_a] #get batch idx where cls token is in a
        cls_in_a_pos = (cls_pos[cls_is_in_a]//2).unsqueeze(1)
        cls_in_b_batch_idx = batch_idx[~cls_is_in_a] #(reverse operator ->switches bool values)
        cls_in_b_pos = (cls_pos[~cls_is_in_a]//2).unsqueeze(1)
        
        # set cls token to -inf, no matter if they are in a or b
        scores[cls_in_a_batch_idx,cls_in_a_pos,:] = -math.inf 
        scores[cls_in_b_batch_idx,:,cls_in_b_pos] = -math.inf
        return scores


    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - 1) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        #take care that the class_token is not being merged
        
            
        scores = set_cls_token_scores_to_mininf(scores)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        #class token position is now not exlpicitly known -> use source to find it later!
            

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    
    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def gather_values_using_batch_indices(data, batch_indices): #adapted include batch dim for source
    """
    Gathers elements from the `data` tensor using the `batch_indices` tensor, allowing different indices for each batch.
    
    Args:
    - batch_indices (torch.Tensor): A 2D tensor of size [batch_size, num_indices] with values in the range [0, num_categories-1].
    - data (torch.Tensor): A tensor of size [batch_size, num_categories, feature_size].
    
    Returns:
    - torch.Tensor: A tensor of size [batch_size, num_indices, feature_size] where each element is selected 
      according to the indices specified in `batch_indices` for each batch.
      
    Used like this: We have indices, where for each index/new token pos, we have the position of (one of) the original tokens
    Additionally, we have data, the values of the new tokens
    We use this to create a blown up version of the data tensor, where we have the current values for all the orig_tokens, wherever their position is now in the current tensor
    """
    device = data.device
    # Ensure batch_indices is a 2D tensor
    assert batch_indices.dim() == 2, "batch_indices must be a 2D tensor"
    
    # Get the dimensions
    batch_size, new_token_num, feature_size = data.size()
    batch_size_idx, orig_token_num = batch_indices.size()

    # Ensure that batch_indices' batch_size matches data's batch_size
    assert batch_size == batch_size_idx, "batch_indices and data must have the same batch_size"
    
    # Ensure that the values in batch_indices are within the range [0, num_categories-1]
    assert torch.all((0 <= batch_indices) & (batch_indices < new_token_num)), "orig_tokens need to map to indices of new_token_num"
    
    # Use advanced indexing to gather data based on the indices for each batch
    # To do this, we need to create an index tensor for the batch dimension
    batch_range = torch.arange(batch_size).unsqueeze(1).to(device)  # Shape: [batch_size, 1]
    
    # Now use batch_indices to index into the data tensor
    gathered_data = data[batch_range, batch_indices, :]  # Shape: [batch_size, num_indices, feature_size]
    assert gathered_data.shape == torch.Size([batch_size, orig_token_num, feature_size]) , "shape is invalid, expected{[batch_size, old_token_num, feature_size]}, got {}"

    return gathered_data

def generate_presence_mask(indices, size): #adapted include batch dim for source
    """
    Generates a boolean mask tensor of specified size based on the indices tensor.

    Args:
    - indices (torch.Tensor): A 2D tensor containing indices (for each batch elem), where each value is in the range [0, size-1].
    - size (int): The size of the output mask tensor.

    Returns:
    - torch.Tensor: A 1D boolean tensor of length `size` where each element is True if its index is present in `indices`, otherwise False.
    """
    device = indices.device
    batch_ids = torch.arange(size[0]).unsqueeze(1).to(device)
    mask = torch.zeros(size, dtype=torch.bool).to(device)  # Initialize the mask tensor with all False values
    mask[batch_ids, indices] = True  # Set True for indices in the source
    return mask


def get_expanded_tokens_and_mask(x: torch.Tensor,source): #adapted include batch dim for source
    """
    takes the current, merged tokens tensor and the source to create a tensor where all the original tokens have their new merged/or unmerged value, no matter where they are now. Additionally give back a mask of orig_token_size where duplicates are False.
    Before entering the Mamba layer, transform the tokens via this function to their original positions with duplicates.
    Transform this row by row flattened version of the picture via the specic pattern.(and the mask)
    After having transformed the token reihenfolge, apply the mask and then feed it into Mamba
    !After the Mamba layer, we need to reorder them to their original position 
    (Can't just adjust the source token, due to multiple patterns per element -> we need to unify the different sequences after each pattern anyways)
    -> For every pattern, batch elem we have own translations? -> How did this work? 
    Idea: AFTER creating the merge fnx, merging tokens AND SOURCE
    use the new source to compute the the original locations of the merged tokens and use the values+the mask to flatten the values&mask appropriate to the wanted pattern and then apply the mask(and reshape if needed)
    """
    batch_size, new_token, old_token_num = source.shape
    new_token_map = source.argmax(1) #Save backtranslation?
    all_original_tokens_corresponding_merged_values = gather_values_using_batch_indices(x,new_token_map)  
    #build on this, 
    token_is_sole_representative_of_group = generate_presence_mask(new_token_map, (batch_size,old_token_num)) 
    return all_original_tokens_corresponding_merged_values, token_is_sole_representative_of_group



# MAP BACK TO PRE PATTERNIZED POSITION FUNCTIONS


def create_map_back(orig_pos_of_tokens_pre_mamba, orig_pos_of_tokens_post_mamba): #adapted include batch dim for source
    """
    Creates the map_back tensor that maps positions in orig_pos_of_tokens_pre_mamba to positions in orig_pos_of_tokens_post_mamba for each batch.
    orig_pos_of_tokens_pre_mamba: 
    orig_pos_of_tokens_post_mamba: 

    Args:
    - orig_pos_of_tokens_pre_mamba (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].
    - orig_pos_of_tokens_post_mamba (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].

    Returns:
    - torch.Tensor: A 2D tensor of size [batch_size, new_token_num] where map_back[i, k] gives the index j such that 
                    orig_pos_of_tokens_post_mamba[i, j] == orig_pos_of_tokens_pre_mamba[i, k] for each batch i.
    """
    # orig_pos_of_tokens_pre_mamba: [batch_size, new_token_num]
    # orig_pos_of_tokens_post_mamba: [batch_size, new_token_num]

    # Broadcasting and matching each element in orig_pos_of_tokens_pre_mamba to elements in orig_pos_of_tokens_post_mamba for each batch
    match_matrix = (orig_pos_of_tokens_pre_mamba.unsqueeze(2) == orig_pos_of_tokens_post_mamba.unsqueeze(1)).int() # [batch_size, new_token_num, new_token_num]

    # Using argmax to find the index where elements match along the new_token_num dimension
    map_back = match_matrix.argmax(dim=2)  # [batch_size, new_token_num]

    return map_back

def transform_post_flattened_tokens_to_position_pre_flatten(x,orig_pos_of_tokens_pre_mamba, orig_pos_of_tokens_post_mamba): #adapted include batch dim for source
    """
    Args:
    - orig_pos_of_tokens_pre_mamba (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].
    - orig_pos_of_tokens_post_mamba (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].
    """
    device = x.device
    batch_size = x.shape[0]
    batch_indices_for_broadcast = torch.arange(batch_size).unsqueeze(1).to(device)
    map_post_mamba_pos_to_pre_mamba_pos = create_map_back(orig_pos_of_tokens_pre_mamba, orig_pos_of_tokens_post_mamba) #ToDo: Parameters change to include batch dim as well
    return x[batch_indices_for_broadcast,map_post_mamba_pos_to_pre_mamba_pos,:]
    # map back: batchsize, num_new_tokens:
    # batch_indices_for_broadcast: batchsize, 1
    # -> for every batch_elem i: choose the indices in map_post_mamba_pos_to_pre_mamba_pos[i]

def repostion_all_tensors_for_mamba_pattern():
    return
# def map_back_vim(x,source):
    


    
