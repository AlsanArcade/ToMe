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
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg

x = torch.tensor([[[1.0, 0.0],   # Token 1
                   [0.0, 10.0],   # Token 2
                   [0.5, 0.5],   # Token 3
                   [0.5, 0.7]]])
merge, _ =  bipartite_soft_matching(
                x,
                1,
                True,
                True,
            )
x = merge(x)
print(f"x new {x}")