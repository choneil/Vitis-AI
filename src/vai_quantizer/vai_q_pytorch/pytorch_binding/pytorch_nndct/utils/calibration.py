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


from nndct_shared.pruning.pruning_lib import PruningSpec
from nndct_shared.nndct_graph import Graph
from nndct_shared.pruning.sensitivity import NetSensitivity
from typing import Mapping, List
from pytorch_nndct.utils import TorchGraphSymbol, logging


def extract_scope_name(node_name: str) -> str:
  return node_name.rsplit(TorchGraphSymbol.NODE_NAME_SEPERATOR, 1)[0]

def extract_scope_name_node_name_map(graph: Graph) -> Mapping[str, List[str]]:
  ret: Mapping[str, str] = {}
  for node in graph.nodes:
    scope_name = extract_scope_name(node.name)
    if scope_name in ret:
      ret[scope_name].append(node.name)
    else:
      ret[scope_name] = [node.name]
  return ret

def calibrate_spec(spec: PruningSpec, graph: Graph) -> PruningSpec:
  scope_name_node_name_map = extract_scope_name_node_name_map(graph)
  for group in spec.groups:
    for idx, node in enumerate(group.nodes):
      scope_name = extract_scope_name(node)
      assert scope_name in scope_name_node_name_map, f"Missing scope_name: '{scope_name}' in graph"
      if len(scope_name_node_name_map[scope_name]) > 0:
        return spec
      new_name = scope_name_node_name_map[scope_name][0]
      group.nodes[idx] = new_name
  return PruningSpec(spec.groups, spec.channel_divisible)

def calibrate_sens(sens: NetSensitivity, graph: Graph) -> NetSensitivity:
  scope_name_node_name_map = extract_scope_name_node_name_map(graph)
  for group in sens.groups:
    for idx, node in enumerate(group.nodes):
      scope_name = extract_scope_name(node)
      assert scope_name in scope_name_node_name_map, f"Missing scope_name: '{scope_name}' in graph"
      if len(scope_name_node_name_map[scope_name]) > 0:
        return sens
      new_name = scope_name_node_name_map[scope_name][0]
      group.nodes[idx] = new_name
  ret = NetSensitivity()
  ret.groups = sens.groups
  ret.uncompleted_steps = ret.uncompleted_steps
  return ret
