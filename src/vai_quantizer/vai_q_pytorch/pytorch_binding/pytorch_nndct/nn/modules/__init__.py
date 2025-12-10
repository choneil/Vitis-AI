# Copyright (C) 2022-2023, Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .conv import *
from .conv1d import *
from .conv3d import *
from .linear import *
from .rnn_builder import *
from .add import *
from .sub import *
from .maxpool import *
from .maxpool1d import *
from .avgpool import *
from .concat import *
from .multiply import *
from .adaptive_avg_pool import *
from .mean import *
from .interpolate import *
from .conv_transpose import *
from .sigmoid import *
from .tanh import *
from .fix_ops import *
from .leaky_relu import *
from .prim_ops import *
from .module_template import *
from .relu import *
from .gelu import *
from .quant_stubs import *
from .reluk import *
from .channel_scale import *
from .function import *
from .hardsigmoid import *
from .hardswish import *
from .quant_noise import *
from .batch_norm import *
from .instance_norm import *
from .group_norm import *
from .correlation1d import *
from .correlation2d import *
from .cost_volume import *
from .softmax import *
from .log_softmax import *
from .layernorm import *
from .embedding import *
from .prelu import *
# from .clamp import *
from .sqrt import *
from pytorch_nndct.utils.torch_utils import CmpFlag, compare_torch_version
if compare_torch_version(CmpFlag.GREATER_EQUAL, "1.9"):
  from .mish import *

from .nndct_quant_model import *
