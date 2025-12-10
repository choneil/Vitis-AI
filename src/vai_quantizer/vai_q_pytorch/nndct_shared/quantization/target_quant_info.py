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



import copy
from collections import OrderedDict
from nndct_shared.nndct_graph import NndctGraphHolder
from nndct_shared.inspector.device_allocator import DPUAllocator
from nndct_shared.inspector.utils import convert_quant_config_to_dict, log_debug_info
from nndct_shared.utils import DeviceType, NNDCT_OP
from nndct_shared.quantization.quant_info import QuantInfoMgr
    

class TargetQuantInfoMgr(QuantInfoMgr):
  def __init__(self, target, graph, model_type, lstm, quant_strategy_info, quant_strategy, custom_quant_ops=None):
    self._device_allocator = DPUAllocator(target)
    super().__init__(graph, model_type, lstm, quant_strategy_info, quant_strategy, custom_quant_ops)
  
  def assign_device_info(self, dev_graph):
    self._device_allocator.process(dev_graph)

  def is_node_quantizable(self, node, lstm):
    if not self._device_allocator._node_device_map:
      return super().is_node_quantizable(node, lstm)
    
    if super().is_node_quantizable(node, lstm):
      if self._device_allocator.is_dpu_node(node):
        return True
      elif node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]:
        if any([self._device_allocator.is_dpu_node(self.Nndctgraph.node(node_name)) for node_name in node.out_nodes]):
          return True
    return False

  def filter_quant_config_by_device_info(self):
    quant_info = copy.deepcopy(self.quant_info)
    for node in self.Nndctgraph.nodes:
      if not self._device_allocator.is_dpu_node(node):
        for param_type, param_tensor in node.op.params.items():
          if param_tensor.name in quant_info["param"]:
            self.quant_info["param"][param_tensor.name] = [None]
            # self.quant_info["param"].pop(param_tensor.name)
            log_debug_info(f"param '{param_tensor} has been removed from quant config")
    quant_nodes = set()
    for node in self.Nndctgraph.nodes:
      if self.is_node_quantizable(node, False):
        quant_nodes.add(self.quant_output(node).name)
        quant_nodes.update([self.quant_output(i_n).name for i_n in node.in_nodes])

    for node_name in quant_info["output"].keys():
      if node_name not in quant_nodes:
        info = self.quant_info["output"][node_name]
        self.quant_info["output"][node_name] = [None] * len(info)
        log_debug_info(f"output '{node_name} has been removed from quant config")
      
    for node_name in quant_info["input"].keys():
      maybe_quantized_nodes = []
      maybe_quantized_nodes += self.quant_groups[node_name]
      if all([not self._device_allocator.is_dpu_node(self.Nndctgraph.node(node)) for node in maybe_quantized_nodes]):
        # self.quant_info["input"].pop(node_name)
        info = self.quant_info["input"][node_name]
        self.quant_info["input"][node_name] = [None] * len(info)
        log_debug_info(f"input '{node_name} has been removed from quant config")
    

      
    