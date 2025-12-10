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
import os

tf.app.flags.DEFINE_string('pb_file', '', 'input pb file')
FLAGS = tf.app.flags.FLAGS

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile(FLAGS.pb_file, "rb").read())
_ = tf.import_graph_def(graphdef, name="")


node_ops = []
for node in graphdef.node:
  if not node.op in node_ops:
    node_ops.append(node.op)

print(node_ops)
