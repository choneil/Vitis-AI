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


import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import os
import argparse
import pdb

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser()
parser.add_argument("graph", type=str, default="", help="tensorflow pb file to load")
FLAGS, uparsed = parser.parse_known_args()

if not gfile.Exists(FLAGS.graph):
    print("Input graph file '"+ FLAGS.input_graph + "' does not exist!")
    print("Usage: python show_nodes graph")
    exit()

with tf.Session() as sess:
    f=tf.gfile.FastGFile(FLAGS.graph,'rb')
    in_graph_def = tf.GraphDef()
    in_graph_def.ParseFromString(f.read())

    for node in in_graph_def.node:
      if node.op in ["DepthwiseConv2dNative","Conv2D"]:
        print(node.name)
      #  if node.op == "Const":
          #  node.attr["value"].tensor.ClearField("tensor_content")
