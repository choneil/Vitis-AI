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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import vai_q_tensorflow as decent_q
# from tensorflow.contrib import decent_q


os.environ["DECENT_DEBUG"] = "0"

src_graph = "./removed_optimized_quantize_model.pb"
dst_graph_dir = "./"
dst_graph = "removed_optimized_quantize_model.pb"


def main():
  name_to_node = {}
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
      tf.graph_util.import_graph_def(src_graph_def, name='')
      dst_graph_def = graph.as_graph_def(add_shapes=True)
    tf.io.write_graph(dst_graph_def, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
