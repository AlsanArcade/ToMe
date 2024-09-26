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


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    cls_token_positionS=None,
    original_cls_token_position: int = 0,
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
        cls_pos = cls_token_positionS  # dim: b, 1
        x = cls_pos.max()
        y = cls_pos.min()
        z = scores.shape
        cls_is_in_a = cls_pos % 2 == 0
        cls_is_in_b = ~cls_is_in_a
        batch_idx = torch.arange(cls_pos.shape[0]).unsqueeze(1).to(device)

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

    def zipTensors(first_tensor, second_tensor, dim_to_zip):
        device = first_tensor.device
        assert len(first_tensor.shape) == len(
            second_tensor.shape
        ), "Tensors must have the same number of dimensions"
        assert (
            abs(first_tensor.shape[dim_to_zip] - second_tensor.shape[dim_to_zip]) <= 1
        ), "The lengths of the dimensions to zip must not differ by more than one"
        assert len(first_tensor.shape) in [2, 3], "len(shape) must be in [2, 3]"
        assert (
            0 <= dim_to_zip < len(first_tensor.shape)
        ), "dim_to_zip must be within valid range"

        if len(first_tensor.shape) == 2:  # Handle 2D tensors
            a, b = first_tensor.shape
            a2, b2 = second_tensor.shape

            if dim_to_zip == 0:
                min_dim_size = min(a, a2)
                result = torch.empty((a + a2, b), dtype=first_tensor.dtype).to(device)
                result[: min_dim_size * 2, :][0::2, :] = first_tensor[:min_dim_size, :]
                result[: min_dim_size * 2, :][1::2, :] = second_tensor[:min_dim_size, :]

                if a > a2:
                    result[min_dim_size * 2 :, :] = first_tensor[min_dim_size:, :]
                elif a2 > a:
                    result[min_dim_size * 2 :, :] = second_tensor[min_dim_size:, :]

            elif dim_to_zip == 1:
                min_dim_size = min(b, b2)
                result = torch.empty((a, b + b2), dtype=first_tensor.dtype).to(device)
                result[:, : min_dim_size * 2][:, 0::2] = first_tensor[:, :min_dim_size]
                result[:, : min_dim_size * 2][:, 1::2] = second_tensor[:, :min_dim_size]

                if b > b2:
                    result[:, min_dim_size * 2 :] = first_tensor[:, min_dim_size:]
                elif b2 > b:
                    result[:, min_dim_size * 2 :] = second_tensor[:, min_dim_size:]

        elif len(first_tensor.shape) == 3:  # Handle 3D tensors
            a, b, c = first_tensor.shape
            a2, b2, c2 = second_tensor.shape

            if dim_to_zip == 0:
                min_dim_size = min(a, a2)
                result = torch.empty((a + a2, b, c), dtype=first_tensor.dtype).to(
                    device
                )
                result[: min_dim_size * 2, :, :][0::2, :, :] = first_tensor[
                    :min_dim_size, :, :
                ]
                result[: min_dim_size * 2, :, :][1::2, :, :] = second_tensor[
                    :min_dim_size, :, :
                ]

                if a > a2:
                    result[min_dim_size * 2 :, :, :] = first_tensor[min_dim_size:, :, :]
                elif a2 > a:
                    result[min_dim_size * 2 :, :, :] = second_tensor[
                        min_dim_size:, :, :
                    ]

            elif dim_to_zip == 1:
                min_dim_size = min(b, b2)
                result = torch.empty((a, b + b2, c), dtype=first_tensor.dtype).to(
                    device
                )
                result[:, : min_dim_size * 2, :][:, 0::2, :] = first_tensor[
                    :, :min_dim_size, :
                ]
                result[:, : min_dim_size * 2, :][:, 1::2, :] = second_tensor[
                    :, :min_dim_size, :
                ]

                if b > b2:
                    result[:, min_dim_size * 2 :, :] = first_tensor[:, min_dim_size:, :]
                elif b2 > b:
                    result[:, min_dim_size * 2 :, :] = second_tensor[
                        :, min_dim_size:, :
                    ]

            elif dim_to_zip == 2:
                min_dim_size = min(c, c2)
                result = torch.empty((a, b, c + c2), dtype=first_tensor.dtype).to(
                    device
                )
                result[:, :, : min_dim_size * 2][:, :, 0::2] = first_tensor[
                    :, :, :min_dim_size
                ]
                result[:, :, : min_dim_size * 2][:, :, 1::2] = second_tensor[
                    :, :, :min_dim_size
                ]

                if c > c2:
                    result[:, :, min_dim_size * 2 :] = first_tensor[:, :, min_dim_size:]
                elif c2 > c:
                    result[:, :, min_dim_size * 2 :] = second_tensor[
                        :, :, min_dim_size:
                    ]

        return result

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r_orig = r
    r = min(r, (t - protected) // 2)
    assert r == r_orig, "cls token position tracking not yet done for this case"

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = (
            metric[..., ::2, :],
            metric[..., 1::2, :],
        )  # vergleiche gerade und ungerade tokens (tokens dim=1)
        scores = a @ b.transpose(-1, -2)  # ähnlichkeit berechnen (scalar)
        # print(f"scores {scores}")
        scores_prev = scores
        scores = set_cls_token_scores_to_mininf(scores)
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # a tokens die nicht gemerged werden

        src_idx = edge_idx[..., :r, :]  # a tokens to merge werden hier ausgewählt

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # ?

    def merge(x: torch.Tensor, mode="mean"):
        device = x.device
        # print(f"merge input: {x.shape}")

        src_original, dst_original = x[..., ::2, :], x[..., 1::2, :]
        b, t1, c = src_original.shape
        _, t2, _ = dst_original.shape

        # Mask of src_size indicationg if the token will remain unmerged
        mask_src_tokens_umnerged = torch.ones((b, t1), dtype=torch.bool).to(device)
        mask_dst_tokens = torch.ones((b, t2), dtype=torch.bool).to(device)

        batch_idx = torch.arange(b).unsqueeze(1).to(device)
        mask_src_tokens_umnerged[batch_idx, src_idx.squeeze(-1)] = False
        # ---------------------------------

        # Mask above blown up to be applied to the zipped tensors
        full_merge_mask = zipTensors(
            mask_src_tokens_umnerged, mask_dst_tokens, dim_to_zip=1
        ).to(device)

        # Merge r src tokens into dst tensor
        src = src_original.gather(dim=-2, index=src_idx.expand(b, r, c))
        dst = dst_original.scatter_reduce(-2, dst_idx.expand(b, r, c), src, reduce=mode)
        _, t2_new, _ = dst.shape
        assert t2_new == t2, "dst dimension should be same after reduce"
        # Combine original src, new dst, remove merged tokens via mask
        temp = zipTensors(src_original, dst, dim_to_zip=1)
        merged_tensor_correct_order = torch.zeros(b, t1 + t2 - r, c).to(device)

        # ----------------------------------
        # Inefficient
        for i in range(b):
            merged_tensor_correct_order[i] = temp[i][full_merge_mask[i], :]
        # Efficient
        # ---------------------------------

        return merged_tensor_correct_order

    def merge_residual(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
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
    # print(f"x size: {x.shape}")
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
        # print(f"Init source size: {source.shape}")
    source = merge(source, mode="amax")
    # print(f"New source size: {source.shape}")

    return source
