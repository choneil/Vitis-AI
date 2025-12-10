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



try:
  import xcompiler
  import xir
except ModuleNotFoundError:
  raise Exception('please install xcompiler package')
from nndct_shared.base import NNDCT_KEYS
from nndct_shared.utils import NndctOption, NndctScreenLogger

class XCompiler(object):

  @staticmethod
  def compile(xmodel_file, target):
    output_file = "_".join([xmodel_file, target]) + NNDCT_KEYS.XMODEL_SUFFIX
    args = ["-i", f"{xmodel_file}_int{NNDCT_KEYS.XMODEL_SUFFIX}", "-o", f"{output_file}", "-t", f"{target}", "--inspector"]
    xcompiler.xcompiler([xmodel_file] + args)
    return output_file

  @classmethod
  def compile_and_reload(cls, xmodel_file, target):
    output_file = cls.compile(xmodel_file, target)
    graph = xir.Graph.deserialize(output_file)
    return graph


  @staticmethod
  def compile_xgraph(xmodel_file, xgraph, target, fingerprint):
    if fingerprint is not None:
      cmd = {
        "inspector": True,
        "fingerprint": [fingerprint],
      }
    elif target is not None:
      cmd = {
        "inspector": True,
        "target": [target],
      }
    
    compiled_graph = xcompiler.xcompiler(xgraph.graph, cmd)
    if NndctOption.nndct_inspect_debug.value:
      output_file = "_".join([xmodel_file, target]) + NNDCT_KEYS.XMODEL_SUFFIX
      compiled_graph.serialize(output_file)
      NndctScreenLogger().info(f"The compiled graph is generated.({output_file})")
    return compiled_graph
