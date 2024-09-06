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

def get_current_cls_token_pos_from_source(original_csl_token_pos,source):
    if(source== None):
        return original_csl_token_pos
    source_flattened = source.squeeze(0)
    new_token_map = source_flattened.argmax(0)
    new_cls_position = new_token_map[original_csl_token_pos]
    return new_cls_position

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    source,
    original_csl_token_pos
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
        cls_pos = get_current_cls_token_pos_from_source(original_csl_token_pos,source)
        cls_in_a = cls_pos%2 == 0
        if cls_in_a:
            scores[:,cls_pos//2,:] = -math.inf
        else:
            scores[:,:,cls_pos//2] = -math.inf
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

def gather_values_using_indices(indices, data):
    """
    Gathers elements from the `data` tensor using the `indices` tensor.
    Args:
    - indices (torch.Tensor): A 1D tensor of size [num_indices] with values in the range [0, num_categories-1].
    - data (torch.Tensor): A tensor of size [batch_size, num_categories, feature_size].
    Used like this: We have indices, where for each index/new token pos, we have the position of (one of) the original tokens
    Additionally, we have data, the values of the new tokens
    We use this to create a blown up version of the data tensor, where we have the current values for all the orig_tokens, wherever their position is now in the current tensor
    
    Returns:
    - torch.Tensor: A tensor of size [batch_size, num_indices, feature_size] where each element is selected 
      according to the indices specified in `indices`.
    """
    # Ensure indices is a 1D tensor
    assert indices.dim() == 1, "indices must be a 1D tensor"
    # Get the dimensions
    batch_size, num_categories, feature_size = data.size()
    num_indices = indices.size(0)
    # Ensure that the values in indices are within the range [0, num_categories-1]
    assert torch.all((0 <= indices) & (indices < num_categories)), "indices contains out-of-range values"
    # Use the indices tensor to index into the data tensor
    # `indices` is of shape [num_indices] so we need to broadcast it across the first dimension (batch_size)
    gathered_data = data[:, indices, :]
    # `gathered_data` should now be of shape [batch_size, num_indices, feature_size]
    return gathered_data
    
def generate_presence_mask(indices, size):
    """
    Generates a boolean mask tensor of specified size based on the indices tensor.

    Args:
    - indices (torch.Tensor): A 1D tensor containing indices, where each value is in the range [0, size-1].
    - size (int): The size of the output mask tensor.

    Returns:
    - torch.Tensor: A 1D boolean tensor of length `size` where each element is True if its index is present in `indices`, otherwise False.
    """
    mask = torch.zeros(size, dtype=torch.bool)  # Initialize the mask tensor with all False values
    mask[indices] = True  # Set True for indices in the source
    return mask


def get_expanded_tokens_and_mask(x: torch.Tensor,source):#INPROGRESS adapted include batch dim for source
    """
    takes the current, merged tokens tensor and the source to create a tensor where all the original tokens have their new merged/or unmerged value, no matter where they are now. Additionally give back a mask of orig_token_size where duplicates are False.
    Idea: AFTER creating the merge fnx, merging tokens AND SOURCE
    use the new source to compute the the original locations of the merged tokens and use the values+the mask to flatten the values&mask appropriate to the wanted pattern and then apply the mask(and reshape if needed)
    """
    new_token, old_token = source.squeeze(0).shape
    new_token_map = source.argmax(0) #Save backtranslation?
    all_original_tokens_corresponding_merged_values = gather_values_using_indices(x,new_token_map)
    #build on this, 
    token_is_sole_representative_of_group = generate_presence_mask(new_token_map, old_token)
    return all_original_tokens_corresponding_merged_values, token_is_sole_representative_of_group



def create_map_back(flat_map, new_tensor_map): #adapted include batch dim for source
    """
    Creates the map_back tensor that maps positions in flat_map to positions in new_tensor_map for each batch.

    Args:
    - flat_map (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].
    - new_tensor_map (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].

    Returns:
    - torch.Tensor: A 2D tensor of size [batch_size, new_token_num] where map_back[i, k] gives the index j such that 
                    new_tensor_map[i, j] == flat_map[i, k] for each batch i.
    """
    # flat_map: [batch_size, new_token_num]
    # new_tensor_map: [batch_size, new_token_num]

    # Broadcasting and matching each element in flat_map to elements in new_tensor_map for each batch
    match_matrix = flat_map.unsqueeze(2) == new_tensor_map.unsqueeze(1)  # [batch_size, new_token_num, new_token_num]

    # Using argmax to find the index where elements match along the new_token_num dimension
    map_back = match_matrix.argmax(dim=2)  # [batch_size, new_token_num]

    return map_back

def transform_post_flattened_tokens_to_position_pre_flatten(x,flat_map, new_tensor_map): #adapted include batch dim for source
    batch_size = x.shape[0]
    batch_indices_for_broadcast = torch.arange(batch_size).unsqueeze(1)
    map_back = create_map_back(flat_map, new_tensor_map) #ToDo: Parameters change to include batch dim as well
    return x[batch_indices_for_broadcast,map_back,:]

def repostion_all_tensors_for_mamba_pattern():
    return
# def map_back_vim(x,source):
    


    
