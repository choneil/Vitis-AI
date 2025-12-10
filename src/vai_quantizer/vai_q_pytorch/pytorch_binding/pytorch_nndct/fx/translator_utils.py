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




def get_meta_info(meta, info_type):
  return getattr(meta, info_type)


def convert_dtype(dtype):
  r"""convert torch dtype to nndct dtype"""
  return {
      'torch.float': 'float32',
      'torch.float32': 'float32',
      'torch.double': 'float64',
      'torch.int': 'int32',
      'torch.long': 'int64'
  }.get(dtype.__str__(), dtype)


def convert_shape(shape):
  return list(shape)