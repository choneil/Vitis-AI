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


class ApproxModes(object):
  NO_APPROX = 'no_approx'
  EXP_POLY = 'exp_poly'
  EXP_LUT = 'exp_lut'
  QIO = 'quant_input_output'

def is_no_approx(mode):
  return mode == ApproxModes.NO_APPROX

def is_exp_poly(mode):
  return mode == ApproxModes.EXP_POLY

def is_exp_lut(mode):
  return mode == ApproxModes.EXP_LUT

def is_quant_input_output(mode):
  return mode == ApproxModes.QIO

def available_modes():
  return [ApproxModes.NO_APPROX, ApproxModes.EXP_POLY, ApproxModes.EXP_LUT]
