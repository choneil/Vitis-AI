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


import os
import sys
from nndct_shared.base import SingletonMeta
from nndct_shared.utils import io, NndctScreenLogger, QError, QWarning, QNote

try:
  import matplotlib.pyplot as plt
except ImportError:
  _enable_plot = False
else:
  _enable_plot = True
  

class Plotter(metaclass=SingletonMeta):
  counter = 0
  figure_dict = {}
  
  def __init__(self):
    if not _enable_plot:
      NndctScreenLogger().warning2user(QWarning.MATPLOTLIB, "Please install matplotlib for visualization.")
      sys.exit(1)
    self._dir = '.nndct_quant_stat_figures'
    io.create_work_dir(self._dir)

  def plot_hist(self, name, data):
    plot_title = "_".join([name, 'hist'])
    if plot_title in self.figure_dict:
      NndctScreenLogger().info("Finish visualization.")
      sys.exit(0)
      
    self.figure_dict[plot_title] = True
    plt.figure(self.counter)
    self.counter += 1
    plt.hist(data, bins=20, facecolor='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title(plot_title)
    plot_title = plot_title.replace('/', '_')
    plt.savefig(os.path.join(self._dir, '.'.join([plot_title, 'svg'])))
    plt.close()
    
    
    
