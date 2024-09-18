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
    cls_token_position: int = 0,
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

    def remove_cls_token_from_tensor(t, cls_token_position):
        cls_token = t[:, cls_token_position, :]
        cls_token_mask = torch.ones(t.size(1), dtype=torch.bool)
        cls_token_mask[cls_token_position] = False  # Set the mask to False at index `i`
        tensor_without_cls_token = t[:, cls_token_mask, :]
        return tensor_without_cls_token, cls_token

    # def zipTensors(first_tensor, second_tensor,dim):
    #     assert len(first_tensor.shape) == len(second_tensor.shape), "tensors must have same amount of dimensions"
    #     assert first_tensor.shape[dim] - second_tensor.shape[dim] <=1, "len of dimensions to zip must not differ by more than one"

    #     assert len(first_tensor.shape) in [2,3], "len(shape) must be in [2,3]
    #     assert dim < len(first_tensor.shape) and dim >0, "dim to zip must be absolute"
    #     if len(first_tensor.shape) == 2:
    #         a,b = first_tensor.shape
    #         a2,b2 = first_tensor.shape
    #         # check which tensor is longer in the dim (if one is), cut off last tensor, zip them along the lines of this:
    #         # result[0::2, :] = first_tensor (for dim == 1)
    #         # result[1::2, :] = second_tensor
    #         # depending on the dim to zip
    #         # return result
    #     # Do this for all possible combinations
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
        metric, _ = remove_cls_token_from_tensor(metric, cls_token_position)

        metric = metric / metric.norm(dim=-1, keepdim=True)
        # wenn token als metric: dims: batch, tokens, emb_dim
        a, b = (
            metric[..., ::2, :],
            metric[..., 1::2, :],
        )  # vergleiche gerade und ungerade tokens (tokens dim=1)
        scores = a @ b.transpose(-1, -2)  # ähnlichkeit berechnen (scalar)
        # print(f"scores {scores}")

        # TESTING LINES
        # scores =  create_tensor_with_ones_at_last_positions(scores)
        # '-----------------

        # wenn token als metric: dims: batch, tokens/2, tokens/2
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(
            dim=-1
        )  # ?für alle token: ähnlichster andrer token id, score
        # print(f"node_max{node_max.shape}")
        # print(f"node_max{node_max}")
        # print(f"node_idx{node_idx.shape}")
        # print(f"node_idx{node_idx}")

        edge_idx = node_max.argsort(dim=-1, descending=True)[
            ..., None
        ]  # absteigend: index der a token mit b token der höchsten ähnlichkeit
        # print(f"edge_idx`n {edge_idx.shape}")
        # print(f"edge_idx`n {edge_idx}")
        # merged_tokens_

        unm_idx = edge_idx[..., r:, :]  # a tokens die nicht gemerged werden
        # print(f"unm_idx`n {unm_idx}")

        src_idx = edge_idx[..., :r, :]  # a tokens to merge werden hier ausgewählt
        # TODO falls reihenfolge rehalten bleiben würde, hier notieren wie viele tokens links weggemerged werden. Ansonsten freestyle und cls token pos tracken only
        # print(f"src_idx`n {src_idx.shape}")
        # print(f"src_idx`n {src_idx}")

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # ?
        # print(f"dst_idx`n {dst_idx.shape}")
        # print(f"dst_idx`n {dst_idx}")

        # if class_token:
        # cls_token_position
    # Need to write this for vim new due to cls token not being at the start
    #         Sort to ensure the class token is at the start
    # unm_idx = unm_idx.sort(dim=1)[0]
    # print(f"unm_idx`n {unm_idx}")

    # def merge_vim_1(x: torch.Tensor, mode="mean"):
    #     # print(f"merge input: {x.shape}")

    #     x, cls_token = remove_cls_token_from_tensor(x,cls_token_position)
    #     src, dst = x[..., ::2, :], x[..., 1::2, :]
    #     b, t1, c = src.shape
    #     _, t2, _ = dst.shape
    #     src_merged_removed = torch.ones(n, t1-r, c)
    #     for i in range(b):
    #         mask = torch.ones(src[0].size(0), dtype=torch.bool)
    #         mask[src_idx.squeeze(-1)[i]] = False# see odd src_idx
    #         x = (src[i])[mask,:]
    #         src_merged_removed[i] = x
    #         #fit the merged tokens right in here at where the original src token was/one of the original tokens was
    #     # Need to reorder unm via unm_idx (opposed to the sorting via scores
    #     # or just use original src with remove/slice src_idx now)
    #     src_merged_removed = src_merged_removed.to('cuda:0')
    #     src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
    #     #arguably doesn't matter here due to src tokens being mixed into dst
    #     dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
    #     #No direct way of knowing the original indexes just from this?
    #     # Need src_idx, dst_idx tensor to locate wanted position of merged token
    #     #Additionally to order: need to keep track of cls token..
    #     t_no_cls =  torch.cat([src_merged_removed, dst], dim=1)
    #     # print(f"cls token shape pre: {cls_token.shape}")
    #     cls_exp = cls_token.unsqueeze(1)
    #     # print(f"cls token shape post unsqueeze: {cls_exp.shape}")

    #     result = torch.cat((t_no_cls[:,:(t1+t2-r)//2,:], cls_exp,t_no_cls[:,(t1+t2-r)//2:,:]), dim=1)
    #     # print(f"result of merge: {result.shape}")
    #     return result, n//2 #cls_pos
    #     # if distill_token:
    #     #     return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
    #     # else:
    #         # return torch.cat([unm, dst], dim=1)

    def merge(x: torch.Tensor, mode="mean"):  # merge_vim_plan_one
        device = x.device
        # print(f"merge input: {x.shape}")

        x, cls_token = remove_cls_token_from_tensor(x, cls_token_position)
        src_original, dst_original = x[..., ::2, :], x[..., 1::2, :]
        b, t1, c = src_original.shape
        _, t2, _ = dst_original.shape

        # Mask of src_size indicationg if the token will remain unmerged
        mask_src_tokens_umnerged = torch.ones((b, t1), dtype=torch.bool).to(device)
        mask_dst_tokens = torch.ones((b, t2), dtype=torch.bool).to(device)

        # ----------------------------------
        # Inefficient
        # for i in range(b):
        #     mask_src_tokens_umnerged[i][src_idx.squeeze(-1)[i]] = False
        # Efficient
        batch_idx = torch.arange(b).unsqueeze(1).to(device)
        mask_src_tokens_umnerged[batch_idx, src_idx.squeeze(-1)] = False
        # ---------------------------------

        # Mask above blown up to be applied to the zipped tensors
        full_merge_mask = zipTensors(
            mask_src_tokens_umnerged, mask_dst_tokens, dim_to_zip=1
        ).to(device)

        # Merge r src tokens into dst tensor
        # src_merged_removed = src_merged_removed.to('cuda:0')
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

        # print(f"cls token shape pre: {cls_token.shape}")
        cls_token_expanded = cls_token.unsqueeze(1)
        new_cls_token_position = (t1 + t2 - r + 1) // 2
        # print(f"cls token shape post unsqueeze: {cls_token_expanded.shape}")

        result = torch.cat(
            (
                merged_tensor_correct_order[:, :new_cls_token_position, :],
                cls_token_expanded,
                merged_tensor_correct_order[:, new_cls_token_position:, :],
            ),
            dim=1,
        )
        # print(f"Merged tensor shape: {result.shape}")
        return result, new_cls_token_position

    # def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
    #     src, dst = x[..., ::2, :], x[..., 1::2, :]
    #     n, t1, c = src.shape
    #     unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
    #     src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
    #     dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

    #     if distill_token:
    #         return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
    #     else:
    #         return torch.cat([unm, dst], dim=1)

    # def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
    #     print_full_tensor=True
    #     src, dst = x[..., ::2, :], x[..., 1::2, :]
    #     if print_full_tensor:
    #         print(f"Initial x tensor:\n{x}")
    #     else:
    #         print(f"Initial x shape: {x.shape}")
    #     if print_full_tensor:
    #         print(f"src tensor:\n{src}")
    #     else:
    #         print(f"src shape: {src.shape}")
    #     if print_full_tensor:
    #         print(f"dst tensor:\n{dst}")
    #     else:
    #         print(f"dst shape: {dst.shape}")
    #     n, t1, c = src.shape
    #     unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
    #     if print_full_tensor:
    #         print(f"unm_idx tensor:\n{unm_idx}")
    #         print(f"unm tensor:\n{unm}")
    #     else:
    #         print(f"unm_idx shape: {unm_idx.shape}")
    #         print(f"unm shape: {unm.shape}")
    #     src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
    #     if print_full_tensor:
    #         print(f"src_idx tensor:\n{src_idx}")
    #         print(f"src (after gather) tensor:\n{src}")
    #     else:
    #         print(f"src_idx shape: {src_idx.shape}")
    #         print(f"src (after gather) shape: {src.shape}")
    #     dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
    #     if print_full_tensor:
    #         print(f"dst_idx tensor:\n{dst_idx}")
    #         print(f"dst (after scatter_reduce) tensor:\n{dst}")
    #     else:
    #         print(f"dst_idx shape: {dst_idx.shape}")
    #         print(f"dst (after scatter_reduce) shape: {dst.shape}")
    #     if distill_token:
    #         result = torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
    #     else:
    #         result = torch.cat([unm, dst], dim=1)
    #     if print_full_tensor:
    #         print(f"Final result tensor:\n{result}")
    #     else:
    #         print(f"Final result shape: {result.shape}")
    # return result

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

    return merge, unmerge, cls_token_position


# def kth_bipartite_soft_matching(
#     metric: torch.Tensor, k: int
# ) -> Tuple[Callable, Callable]:
#     """
#     Applies ToMe with the two sets as (every kth element, the rest).
#     If n is the number of tokens, resulting number of tokens will be n // z.

#     Input size is [batch, tokens, channels].
#     z indicates the stride for the first set.
#     z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
#     """
#     if k <= 1:
#         return do_nothing, do_nothing

#     def split(x):
#         t_rnd = (x.shape[1] // k) * k
#         x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
#         a, b = (
#             x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
#             x[:, :, (k - 1), :],
#         )
#         return a, b

#     with torch.no_grad():
#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = split(metric)
#         r = a.shape[1]
#         scores = a @ b.transpose(-1, -2)

#         _, dst_idx = scores.max(dim=-1)
#         dst_idx = dst_idx[..., None]

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = split(x)
#         n, _, c = src.shape
#         dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

#         return dst

#     def unmerge(x: torch.Tensor) -> torch.Tensor:
#         n, _, c = x.shape
#         dst = x

#         src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

#         src = src.view(n, -1, (k - 1), c)
#         dst = dst.view(n, -1, 1, c)

#         out = torch.cat([src, dst], dim=-2)
#         out = out.contiguous().view(n, -1, c)

#         return out

#     return merge, unmerge


# def random_bipartite_soft_matching(
#     metric: torch.Tensor, r: int
# ) -> Tuple[Callable, Callable]:
#     """
#     Applies ToMe with the two sets as (r chosen randomly, the rest).
#     Input size is [batch, tokens, channels].

#     This will reduce the number of tokens by r.
#     """
#     if r <= 0:
#         return do_nothing, do_nothing

#     with torch.no_grad():
#         B, N, _ = metric.shape
#         rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

#         a_idx = rand_idx[:, :r, :]
#         b_idx = rand_idx[:, r:, :]

#         def split(x):
#             C = x.shape[-1]
#             a = x.gather(dim=1, index=a_idx.expand(B, r, C))
#             b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
#             return a, b

#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = split(metric)
#         scores = a @ b.transpose(-1, -2)

#         _, dst_idx = scores.max(dim=-1)
#         dst_idx = dst_idx[..., None]

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = split(x)
#         C = src.shape[-1]
#         dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

#         return dst

#     def unmerge(x: torch.Tensor) -> torch.Tensor:
#         C = x.shape[-1]
#         dst = x
#         src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

#         out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

#         out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
#         out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

#         return out

#     return merge, unmerge


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
    # """
    # For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    # x is used to find out how many tokens there are in case the source is None.
    # """
    # # print(f"x size: {x.shape}")
    # if source is None:
    #     n, t, _ = x.shape
    #     source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    #     # print(f"Init source size: {source.shape}")
    # source = merge(source, mode="amax")
    # # print(f"New source size: {source.shape}")

    return source
