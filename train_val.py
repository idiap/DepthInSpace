# DepthInSpace is a PyTorch-based program which estimates 3D depth maps
# from active structured-light sensor's multiple video frames.
#
# MIT License
#
# Copyright, 2021 ams International AG
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#  
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from model import single_frame_worker
from model import multi_frame_worker
from model import networks
from model import multi_frame_networks
from co.args import parse_args

torch.backends.cudnn.benchmark = True

# parse args
args = parse_args()

# loss types
if args.architecture == 'single_frame':
  worker = single_frame_worker.Worker(args)
elif args.architecture == 'multi_frame':
  worker = multi_frame_worker.Worker(args)

if args.use_pseudo_gt and args.architecture != 'single_frame':
  print("Using pseudo-gt is only possible in single-frame architecture")
  raise NotImplementedError

# set up network
if args.architecture == 'single_frame':
  net = networks.DispDecoder(channels_in=2, max_disp=args.max_disp, imsizes=worker.imsizes)
elif args.architecture == 'multi_frame':
  net = multi_frame_networks.FuseNet(imsize=worker.imsizes[0], K=worker.K, baseline=worker.baseline, track_length=worker.track_length, max_disp=args.max_disp)

# optimizer
opt_parameters = net.parameters()
optimizer = torch.optim.Adam(opt_parameters, lr=1e-4)

# start the work
worker.do(net, optimizer)