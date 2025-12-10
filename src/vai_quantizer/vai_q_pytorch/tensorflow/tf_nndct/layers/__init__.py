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


"""Nndct layers API."""

from tf_nndct.utils import tf_utils
if tf_utils.is_tf_version_less_than('2.9.0'):
  from tf_nndct.layers.recurrent import LSTM
  from tf_nndct.layers.recurrent import LSTMCell
else:
  from keras.layers import LSTM
  from keras.layers import LSTMCell

from tf_nndct.layers.array import Identity

from tf_nndct.layers.math import Dense
from tf_nndct.layers.math import Add
from tf_nndct.layers.math import Multiply
from tf_nndct.layers.math import Sigmoid
from tf_nndct.layers.math import Tanh
