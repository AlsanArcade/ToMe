# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

# Mehrere wrapper-> mamba patchen, letzte mlp wird aus der forward funktion genommen, 
# dann ein wrapper, der die neue mamba mit einer anderen klasse, welche das forward enthält, in einer forward verbindet
# dann in mamba die mamba_inner_fn_no_out_proj funktion anpassen, dass für den merge relevante dinge zurückgegeben werden
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple

import torch

from mamba_ssm.modules.mamba_simple import Mamba
import sys
from vim import models_mamba

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r


class ToMeBlock(models_mamba.Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
        if residual is None:
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        else:
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )    
        x_0 = hidden_states.shape
        y_0 = residual.shape
        # rearrange here !!!
        hidden_states, metric = self.mixer(hidden_states, inference_params=inference_params)
        # rearrange back here
        
        

        r = self._tome_info["r"].pop(0)
        print(f"hidden size before token red {hidden_states.shape}")
        if r > 0:
            hidden_states = hidden_states[:, :-1, :]
            residual = residual[:, :-1, :]
        print(f"hidden size after token red {hidden_states.shape}")
        
        hidden_states = F.linear(hidden_states, self.out_proj_weight, self.out_proj_bias)
        x_1 = hidden_states.shape
        y_1 = residual.shape
        # print(self._tome_info["source"])
        return hidden_states, residual
    # def reorderForMambaBlock(hidden_states, source):
    #     #create full dimensional hidden states from current hidden states, source
    #     #create mask from source (see visualization ipynb?)
    #     # flatten full dim hidden, mask (myb make list of tensors -one dim ->list), map 
    #     # delete all duplicate tensors via mask, shorten tensor (?)
    #     return
    # def orderBackToOriginal(hidden_states, source):
    #     return


class ToMeMamba(Mamba):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        x_001 = hidden_states.shape
        # print(f"hidden size start{hidden_states.shape}")

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                print("step")
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # x and z ! in one ll, then split later: x, z = xz.chunk(2, dim=1)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        # print(f"hidden size after rearrange, proj:{xz.shape}")
        
        x_002 = xz.shape
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        x_003 = xz.shape

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            A_b = -torch.exp(self.A_b_log.float())
            out = mamba_inner_fn_no_out_proj(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out_b = mamba_inner_fn_no_out_proj(
                xz.flip([-1]),
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_b,
                None,
                None,
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            out_mixed = rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2
            z = out.shape
            z1 = out_b.shape
            z3 = out_mixed.shape

            metric = out_mixed # oder einfach (out + out_b.flip([-1]) ?
            
        else:
            assert False
        if self.init_layer_scale is not None:
                out = out * self.gamma    
        # return out
        return out_mixed, metric



def make_tome_class(transformer_class):
    class ToMeVisionMamba(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.layers), self.r) 
            self._tome_info["size"] = None
            self._tome_info["size_residual"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionMamba


def apply_patch(
    model: models_mamba.VisionMamba, trace_source: bool = True, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionMamba = make_tome_class(model.__class__)
    
    model.__class__ = ToMeVisionMamba
    model.r = 1#(1, -1.0) # 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "size_residual": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    out_proj_weights = []
    out_proj_biases = []
    for module in model.modules():
        if isinstance(module, Mamba):
            module.__class__ = ToMeMamba
            out_proj_weights.append(module.out_proj.weight)
            out_proj_biases.append(module.out_proj.bias)
    print(f"len out proj list{len(out_proj_weights)}")
    i = 0
    for module in model.modules():
        if isinstance(module, models_mamba.Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
            module.out_proj_weight = out_proj_weights[i]
            module.out_proj_bias = out_proj_biases[i]
            i += 1
    print(f"i value, should be same{i}")
    
    x = "I'm a breakpoint, yeah!"

# def reorderMergedTokensToKeepFlattenedOrder():
    
    
