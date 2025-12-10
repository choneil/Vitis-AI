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


import numpy as np
import pickle
with open('./test/org_w.pickle', 'rb') as handle:
  org_w = pickle.load(handle)
with open('./test/org_b.pickle', 'rb') as handle:
  org_b = pickle.load(handle)
with open('./test/offset.pickle', 'rb') as handle:
  offset = pickle.load(handle)
with open('./test/scale.pickle', 'rb') as handle:
  scale = pickle.load(handle)


with open('./test/fold_w.pickle', 'rb') as handle:
  fold_w = pickle.load(handle)
with open('./test/fold_b.pickle', 'rb') as handle:
  fold_b = pickle.load(handle)

new_w = np.zeros_like(org_w)
new_b = np.zeros_like(org_b)
h, w, out, in_num = new_w.shape
for i in range(out):
  for l in range(in_num):
    for j in range(h):
      for k in range(w):
        new_w[j,k,i,l] = org_w[j,k,i,l] * scale[i]
new_b = scale * org_b + offset

print(np.max(np.abs(new_w-fold_w)))
print(np.max(np.abs(new_b-fold_b)))
import pdb; pdb.set_trace()
print(new_w.shape)
