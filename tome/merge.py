# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def select_specific_tokens_per_batch_3d(data, tokens_to_select_per_batch):
    """
    Per batch, the tokens that are listed in tokens_to_select_per_batch[batch] are selected.
    tokens_to_select_per_batch[batch].shape[0] can be smaller, equal, or larger than data[batch].shape[0]
    <> You can select every token zero, one or multiple times
    """
    assert (
        len(tokens_to_select_per_batch.shape) == 2
    ), "tokens to select must be a 2d tensor, if there is only one token per batch, pass it as single element list"
    assert len(data.shape) == 3, "input data needs to be 3d"
    assert (
        tokens_to_select_per_batch.shape[0] == data.shape[0]
    ), "batch size must be same"
    assert torch.all(
        (0 <= tokens_to_select_per_batch) & (tokens_to_select_per_batch < data.shape[1])
    ), "tokens to choose must be valid token indices of data"
    device = data.device
    batch_size = data.shape[0]
    batch_id_for_join = torch.arange(batch_size).unsqueeze(1).to(device)
    data_of_selected_tokens = data[batch_id_for_join, tokens_to_select_per_batch, :]
    return data_of_selected_tokens


def select_specific_tokens_per_batch_2d(data, tokens_to_select_per_batch):
    assert (
        len(tokens_to_select_per_batch.shape) == 2
    ), "tokens to select must be a 2d tensor, if there is only one token per batch, pass it as single element list"
    assert len(data.shape) == 2, "input data needs to be 3d"
    assert (
        tokens_to_select_per_batch.shape[0] == data.shape[0]
    ), "batch size must be same"
    device = data.device
    batch_size = data.shape[0]
    batch_id_for_join = torch.arange(batch_size).unsqueeze(1).to(device)
    data_of_selected_tokens = data[batch_id_for_join, tokens_to_select_per_batch]
    return data_of_selected_tokens


def do_nothing(x, mode=None):
    return x


def get_current_cls_token_pos_from_source(original_cls_token_positions, source):
    if source == None:
        return original_cls_token_positions
    orig_pos = original_cls_token_positions[0]  # always same for each batch elem
    new_pos_of_orig_tokens = source.argmax(1)
    cl_pos_new = new_pos_of_orig_tokens[:, orig_pos]

    return cl_pos_new


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    source,
    original_cls_token_positions,  # original_cls_token_positions
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    """

    def set_cls_token_scores_to_mininf(scores):
        device = scores.device
        cls_pos = get_current_cls_token_pos_from_source(
            original_cls_token_positions, source
        )  # dim: b, 1
        x = cls_pos.max()
        y = cls_pos.min()
        z = scores.shape
        cls_is_in_a = cls_pos % 2 == 0
        cls_is_in_b = ~cls_is_in_a
        batch_idx = (
            torch.arange(original_cls_token_positions.shape[0]).unsqueeze(1).to(device)
        )

        cls_in_a_batch_idx = batch_idx[
            cls_is_in_a
        ]  # get batch idx where cls token is in a
        if len(cls_in_a_batch_idx > 0):
            cls_pos_that_will_end_up_in_a = cls_pos[cls_is_in_a]
            cls_pos_in_a = cls_pos_that_will_end_up_in_a // 2
            cls_in_a_pos_as_tokens_per_batch = cls_pos_in_a.unsqueeze(1)
            scores[cls_in_a_batch_idx, cls_in_a_pos_as_tokens_per_batch, :] = (
                -math.inf
            )  # set similarity scores with cls token to -inf to avoind merging

        cls_in_b_batch_idx = batch_idx[
            cls_is_in_b
        ]  # get batch idx where cls token is in a
        if len(cls_in_b_batch_idx > 0):
            cls_pos_that_will_end_up_in_b = cls_pos[cls_is_in_b]
            cls_pos_in_b = cls_pos_that_will_end_up_in_b // 2
            cls_in_b_pos_as_tokens_per_batch = cls_pos_in_b.unsqueeze(1)
            scores[cls_in_b_batch_idx, :, cls_in_b_pos_as_tokens_per_batch] = (
                -math.inf
            )  # set similarity scores with cls token to -inf to avoid merging

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

        # take care that the class_token is not being merged
        scores = set_cls_token_scores_to_mininf(scores)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        # class token position is now not explicitly known -> use source to find it later!

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # print(f"unm_idx\n{unm_idx}")
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # print(f"src_idx\n{src_idx}")
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


def return_mask_marking_specified_token_indices_per_batch(indices, size):
    """
    Takes the values of indices and creates a mask of size size where mask[i,j] is only set to true if indices[i] contains the value j (at any position indices[i][x])
    """
    device = indices.device
    batch_ids = torch.arange(size[0]).unsqueeze(1).to(device)
    mask = torch.zeros(size, dtype=torch.bool).to(
        device
    )  # Initialize the mask tensor with all False values
    mask[batch_ids, indices] = True  # Set True for indices in the source
    return mask


def get_expanded_tokens_and_mask(
    x: torch.Tensor, source
):  # adapted include batch dim for source
    """
    takes the current, merged tokens tensor and the source to create a tensor where all the original tokens have their new merged/or unmerged value, no matter where they are now.
    Additionally give back a mask of orig_token_size where duplicat tensors are set to False (so for each merged token, only one of the original will be set to true).
    Notes for usage:
    Before entering the Mamba layer, transform the tokens via this function to their original positions with duplicates.
    Transform this row by row flattened version of the picture via the specic pattern.(and the mask)
    After having transformed the token order, apply the mask and then feed it into Mamba
    !After the Mamba layer, we need to reorder them to their original position
    (Can't just adjust the source token, due to multiple patterns per element -> we need to unify the different sequences after each pattern anyways)
    -> For every pattern, batch elem we have own translations? -> How did this work?
    Idea: AFTER creating the merge fnx, merging tokens AND SOURCE
    use the new source to compute the the original locations of the merged tokens and use the values+the mask to flatten the values&mask appropriate to the wanted pattern and then apply the mask(and reshape if needed)
    """
    batch_size, new_token_num, old_token_num = source.shape
    new_pos_of_every_orig_token = source.argmax(1)  # Save backtranslation?
    all_original_tokens_corresponding_merged_values = (
        select_specific_tokens_per_batch_3d(x, new_pos_of_every_orig_token)
    )
    token_is_sole_representative_of_group = (
        return_mask_marking_specified_token_indices_per_batch(
            new_pos_of_every_orig_token, size=(batch_size, old_token_num)
        )
    )
    return (
        all_original_tokens_corresponding_merged_values,
        token_is_sole_representative_of_group,
    )


# MAP BACK TO PRE PATTERNIZED POSITION FUNCTIONS


def create_map_back(
    orig_pos_of_tokens_pre_mamba, orig_pos_of_tokens_post_mamba
):  # adapted include batch dim for source
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
    match_matrix = (
        orig_pos_of_tokens_pre_mamba.unsqueeze(2)
        == orig_pos_of_tokens_post_mamba.unsqueeze(1)
    ).int()  # [batch_size, new_token_num, new_token_num]

    # Using argmax to find the index where elements match along the new_token_num dimension
    map_back = match_matrix.argmax(dim=2)  # [batch_size, new_token_num]

    return map_back


def transform_post_flattened_tokens_to_position_pre_flatten(
    x, orig_pos_of_tokens_pre_mamba, orig_pos_of_tokens_post_mamba
):  # adapted include batch dim for source
    """
    Args:
    - orig_pos_of_tokens_pre_mamba (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].
    - orig_pos_of_tokens_post_mamba (torch.Tensor): A 2D tensor of size [batch_size, new_token_num] with elements in the range [0, orig_token_num-1].
    """
    device = x.device
    batch_size = x.shape[0]
    batch_indices_for_broadcast = torch.arange(batch_size).unsqueeze(1).to(device)
    map_post_mamba_pos_to_pre_mamba_pos = create_map_back(
        orig_pos_of_tokens_pre_mamba, orig_pos_of_tokens_post_mamba
    )
    return x[batch_indices_for_broadcast, map_post_mamba_pos_to_pre_mamba_pos, :]
    # map back: batchsize, num_new_tokens:
    # batch_indices_for_broadcast: batchsize, 1
    # -> for every batch_elem i: choose the indices in map_post_mamba_pos_to_pre_mamba_pos[i]


def repostion_all_tensors_for_mamba_pattern():
    return


# def map_back_vim(x,source):
