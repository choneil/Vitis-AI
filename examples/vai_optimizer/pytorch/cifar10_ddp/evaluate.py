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


import argparse
import os

import torch

from data import get_dataloader
from net import MyNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
parser.add_argument(
    '--pretrained',
    type=str,
    default='./baseline_bak/model.pth',
    help='Pretrained model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./dataset/cifar10',
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=64,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

args, _ = parser.parse_known_args()


device = 'cuda'
gpus = get_gpus(args.gpus)

if __name__ == '__main__':
  model = MyNet()
  model = torch.nn.DataParallel(model, device_ids=gpus)
  batch_size = args.batch_size * len(gpus)

  if os.path.exists(args.data_dir):
    download = False
  else:
    download = True

  val_loader = get_dataloader(args.data_dir, batch_size, num_workers=args.num_workers, shuffle=False, train=False, download=download)

  model.to(device)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  if not os.path.exists(args.pretrained):
    print('Pretrained model do not exist!')
  model.load_state_dict(torch.load(args.pretrained))
  acc1, acc5 = evaluate(val_loader, model, criterion)
  print('acc1={},acc5={}'.format(acc1, acc5))
