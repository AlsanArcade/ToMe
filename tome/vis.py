# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms

from PIL import Image

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


# def make_visualization(
#     img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
# ) -> Image:
#     """
#     Create a visualization like in the paper.

#     Args:
#      -

#     Returns:
#      - A PIL image the same size as the input.
#     """

#     img = np.array(img.convert("RGB")) / 255.0
#     source = source.detach().cpu()

#     h, w, _ = img.shape
#     print(img.shape)

#     ph = h // patch_size
#     pw = w // patch_size
#     print(ph)
#     print(pw)
#     if class_token:
#         source = source[:, :, 1:]

#     vis = source.argmax(dim=1)
#     num_groups = vis.max().item() + 1

#     cmap = generate_colormap(num_groups)
#     vis_img = 0

#     for i in range(num_groups):
#         mask = (vis == i).float().view(1, 1, ph, pw)
#         mask = F.interpolate(mask, size=(h, w), mode="nearest")
#         mask = mask.view(h, w, 1).numpy()

#         color = (mask * img).sum(axis=(0, 1)) / mask.sum()
#         mask_eroded = binary_erosion(mask[..., 0])[..., None]
#         mask_edge = mask - mask_eroded

#         if not np.isfinite(color).all():
#             color = np.zeros(3)

#         vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
#         vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

#     # Convert back into a PIL image
#     vis_img = Image.fromarray(np.uint8(vis_img * 255))

#     return vis_img


def make_visualization(
    img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     - img: Input image
     - source: Source tensor from the model
     - patch_size: Size of the patches
     - class_token: Whether to include the class token

    Returns:
     - A PIL image the same size as the input.
    """

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    print(f"Image shape: {img.shape}")

    ph = h // patch_size
    pw = w // patch_size
    print(f"ph: {ph}, pw: {pw}")

    if class_token:
        print(f"Source shape before class token removal: {source.shape}")
        source = source[:, :, 1:]
        print(f"Source shape after class token removal: {source.shape}")

    vis = source.argmax(dim=1)
    print(f"Vis shape: {vis.shape}")
    num_groups = vis.max().item() + 1
    print(f"Number of groups: {num_groups}")

    cmap = generate_colormap(num_groups)
    vis_img = np.zeros((h, w, 3))

    for i in range(num_groups):
        mask = (vis == i).float()
        print(f"Mask shape before view: {mask.shape}")
        mask = mask.view(1, 1, ph, pw)
        print(f"Mask shape after view: {mask.shape}")
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img


def make_visualization_mamba(
    img: Image, source: torch.Tensor, token_per_dim, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     - img: Input image
     - source: Source tensor from the model
     - patch_size: Size of the patches
     - class_token: Whether to include the class token

    Returns:
     - A PIL image the same size as the input.
    """

    # CenterCrop image to enable visualization of the tokens generated from the image
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(216),
    ])
    img = transform(img)
    img = transforms.functional.to_pil_image(img)

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()



    h, w, _ = img.shape
    print(f"Image shape: {img.shape}")

    ph = token_per_dim
    pw = token_per_dim
    print(f"ph: {ph}, pw: {pw}")

    if class_token:
        print(f"Source shape before class token removal: {source.shape}")
        source = source[:, :, 1:]
        print(f"Source shape after class token removal: {source.shape}")

    vis = source.argmax(dim=1)
    print(f"Vis shape: {vis.shape}")
    num_groups = vis.max().item() + 1
    print(f"Number of groups: {num_groups}")

    cmap = generate_colormap(num_groups)
    vis_img = np.zeros((h, w, 3))

    for i in range(num_groups):
        mask = (vis == i).float()
        mask = mask.view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img

def make_visualization_mamba_only_merged_tokens(
        
    img: Image, source: torch.Tensor, token_per_dim, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     - img: Input image
     - source: Source tensor from the model
     - patch_size: Size of the patches
     - class_token: Whether to include the class token

    Returns:
     - A PIL image the same size as the input.
    """

    # CenterCrop image to enable visualization of the tokens generated from the image
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(216),
    ])
    img = transform(img)
    img = transforms.functional.to_pil_image(img)

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()



    h, w, _ = img.shape
    print(f"Image shape: {img.shape}")

    ph = token_per_dim
    pw = token_per_dim
    print(f"ph: {ph}, pw: {pw}")

    if class_token:
        print(f"Source shape before class token removal: {source.shape}")
        source = source[:, :, 1:]
        print(f"Source shape after class token removal: {source.shape}")

    print(f"Source shape after class token removal: {source.shape}")

    sum_of_original_tokens_per_new_token = source.sum(dim=2)
    print(f"sum_of_original_tokens_per_new_token: {sum_of_original_tokens_per_new_token.shape}")

    first_token_got_merged = sum_of_original_tokens_per_new_token[0][0]>1
    print(f"sum_of_merged_tokens: {sum_of_original_tokens_per_new_token}")

    mask_multiple_tokens = sum_of_original_tokens_per_new_token > 1
    num_merged_tokens = sum(mask_multiple_tokens[0].int())
    print(f"num_merged_tokens: {num_merged_tokens}")
    print(f"mask_multiple_tokens: {mask_multiple_tokens.shape}")
    mask_multiple_tokens_expanded = mask_multiple_tokens.unsqueeze(-1).expand_as(source)
    print(f"mask_multiple_tokens_expanded: {mask_multiple_tokens_expanded.shape}")
    source_without_single_token_adjacency =  source * mask_multiple_tokens_expanded
    print(f"source_without_single_token_adjacency: {source_without_single_token_adjacency.shape}")

    vis = source_without_single_token_adjacency.argmax(dim=1)
    print(f"vis: {vis.shape}")
    vis_original_for_first_group = source.argmax(dim=1)
    print(f"vis_original_for_first_group: {vis_original_for_first_group.shape}")
 #add extra dim, then "create" adjency to this new "orig_token", then ignore this "group" later in the loop
    print(f"Vis : {vis}")
    print(f"Vis shape: {vis.shape}")
    groups = set(vis[0])
    print(f"Number of groups as per groups size: {len(groups)}")
    num_groups = len(groups)
    print(f"Number of merged tokens: {num_merged_tokens}")

    cmap = generate_colormap(num_groups)
    vis_img = np.zeros((h, w, 3))

    for i in range(num_groups):
        if i == 0 & first_token_got_merged:
            mask = (vis_original_for_first_group == i).float()
        else:
            mask = (vis == i).float()
        # mask = (vis == i-1).float()
        print(f"Mask shape before view: {mask.shape}")
        print(f"Mask sum: {mask.sum()}")
        if mask.sum()<=0:
            print(f"Mask sum ZERO)
        mask = mask.view(1, 1, ph, pw)
        # print(f"Mask shape after view: {mask.shape}")
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img


# def make_visualization_mamba_only_merged_tokens(def make_visualization_mamba_only_merged_tokens(
        
#     img: Image, source: torch.Tensor, token_per_dim, class_token: bool = True
# ) -> Image:
#     """
#     Create a visualization like in the paper.

#     Args:
#      - img: Input image
#      - source: Source tensor from the model
#      - patch_size: Size of the patches
#      - class_token: Whether to include the class token

#     Returns:
#      - A PIL image the same size as the input.
#     """

#     # CenterCrop image to enable visualization of the tokens generated from the image
#     transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.CenterCrop(216),
#     ])
#     img = transform(img)
#     img = transforms.functional.to_pil_image(img)

#     img = np.array(img.convert("RGB")) / 255.0
#     source = source.detach().cpu()



#     h, w, _ = img.shape
#     print(f"Image shape: {img.shape}")

#     ph = token_per_dim
#     pw = token_per_dim
#     print(f"ph: {ph}, pw: {pw}")

#     if class_token:
#         print(f"Source shape before class token removal: {source.shape}")
#         source = source[:, :, 1:]
#         print(f"Source shape after class token removal: {source.shape}")

#     print(f"Source shape after class token removal: {source.shape}")

#     sum_of_original_tokens_per_new_token = source.sum(dim=1)
#     print(f"sum_of_merged_tokens: {sum_of_original_tokens_per_new_token}")

#     mask = sum_of_original_tokens_per_new_token < 2
#     print(f"mask: {mask}")

#     vis = source.argmax(dim=1)
#     print(f"Vis : {vis}")
#     print(f"Vis shape: {vis.shape}")
#     vis_merged = torch.where(mask, torch.full_like(vis, -1), vis)
#     num_groups = mask.int().sum(dim=1)
#     print(f"Number of groups: {num_groups}")

#     cmap = generate_colormap(num_groups)
#     vis_img = np.zeros((h, w, 3))

#     for i in range(num_groups):
#         mask = (vis_merged == i).float()
#         # mask = (vis == i-1).float()
#         # print(f"Mask shape before view: {mask.shape}")
#         mask = mask.view(1, 1, ph, pw)
#         # print(f"Mask shape after view: {mask.shape}")
#         mask = F.interpolate(mask, size=(h, w), mode="nearest")
#         mask = mask.view(h, w, 1).numpy()

#         color = (mask * img).sum(axis=(0, 1)) / mask.sum()
#         mask_eroded = binary_erosion(mask[..., 0])[..., None]
#         mask_edge = mask - mask_eroded

#         if not np.isfinite(color).all():
#             color = np.zeros(3)

#         vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
#         vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

#     # Convert back into a PIL image
#     vis_img = Image.fromarray(np.uint8(vis_img * 255))

#     return vis_img