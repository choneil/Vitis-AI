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


from pytorch_nndct.nn.quantization.modules.activation import DPULeakyReLU
from pytorch_nndct.nn.quantization.modules.activation import GELU
from pytorch_nndct.nn.quantization.modules.activation import Sigmoid
from pytorch_nndct.nn.quantization.modules.activation import Softmax
from pytorch_nndct.nn.quantization.modules.activation import Tanh
from pytorch_nndct.nn.quantization.modules.batchnorm import QuantizedBatchNorm2d
from pytorch_nndct.nn.quantization.modules.batchnorm import QuantizedBatchNorm3d
from pytorch_nndct.nn.quantization.modules.bfp import BFPQuantizer
from pytorch_nndct.nn.quantization.modules.bfp import BFPPrimeQuantizer
from pytorch_nndct.nn.quantization.modules.bfp import BFPPrimeSharedQuantizer
from pytorch_nndct.nn.quantization.modules.conv import QuantizedConv1d
from pytorch_nndct.nn.quantization.modules.conv import QuantizedConv2d
from pytorch_nndct.nn.quantization.modules.conv import QuantizedConv3d
from pytorch_nndct.nn.quantization.modules.conv import QuantizedConvTranspose2d
from pytorch_nndct.nn.quantization.modules.conv import QuantizedConvTranspose3d
from pytorch_nndct.nn.quantization.modules.conv_fused import QuantizedConvBatchNorm2d
from pytorch_nndct.nn.quantization.modules.conv_fused import QuantizedConvBatchNorm3d
from pytorch_nndct.nn.quantization.modules.conv_fused import QuantizedConvTransposeBatchNorm2d
from pytorch_nndct.nn.quantization.modules.conv_fused import QuantizedConvTransposeBatchNorm3d
from pytorch_nndct.nn.quantization.modules.fp import BFloat16Quantizer
from pytorch_nndct.nn.quantization.modules.fp import FP32Quantizer
from pytorch_nndct.nn.quantization.modules.fp import FP8Quantizer
from pytorch_nndct.nn.quantization.modules.linear import QuantizedLinear
from pytorch_nndct.nn.quantization.modules.normalization import LayerNorm
from pytorch_nndct.nn.quantization.modules.pooling import DPUAdaptiveAvgPool2d
from pytorch_nndct.nn.quantization.modules.pooling import DPUAvgPool2d
from pytorch_nndct.nn.quantization.modules.tqt import TQTQuantizer
