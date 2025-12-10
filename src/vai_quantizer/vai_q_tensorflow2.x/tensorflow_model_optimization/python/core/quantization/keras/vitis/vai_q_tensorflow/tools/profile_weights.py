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
import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

src_graph = "./optimized.pb"
dst_graph_dir = "./"
dst_graph = "removed_quantize_eval_model.pb"

def draw_boxplot(data, title):
  fig1, ax1 = plt.subplots()
  ax1.set_title(title)
  ax1.boxplot(data)
  title = title.replace("/", "_")
  fig1.savefig("./plots/" + title + ".png")
  plt.close()


def main():
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())

  head, tail = False, False
  for node in src_graph_def.node:
    if node.name.endswith("/weights"):
      w = tensor_util.MakeNdarray(node.attr['value'].tensor)
      print(node.name, w.shape)
      _, _, _, k_out = w.shape
      w = w.reshape((-1, k_out))
      # w_max_1 = np.max(np.abs(w), axis=(0, 1, 2))
      # w_max_2 = np.max(np.abs(w), axis=(0, 1, 3))
      # print(w_max_1)
      # print(w_max_2)
      draw_boxplot([w[:,d] for d in range(k_out)], node.name)
      # import pdb; pdb.set_trace()

    if node.name.endswith("/biases"):
      b = tensor_util.MakeNdarray(node.attr['value'].tensor)
      print(node.name, b.shape)
      # b_max = np.max(np.abs(b), axis=0)
      # print(b)
      draw_boxplot(b, node.name)
      # import pdb; pdb.set_trace()

if __name__ == '__main__':
  main()
