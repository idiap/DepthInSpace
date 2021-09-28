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

import numpy as np
from pathlib import Path
import os
import sys
import argparse
import json
import pickle
import h5py
import torch
from tqdm import tqdm

sys.path.append('../')
from model import networks
from model import multi_frame_networks

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('architecture',
                        help='Select the architecture to produce disparity (single_frame/multi_frame)',
                        choices=['single_frame', 'multi_frame'], type=str)
    parser.add_argument('--epoch',
                        help='Epoch whose results will be pre-saved',
                        default=-1, type=int)
    args = parser.parse_args()

    # output directory
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    with open(config_path) as fp:
        config = json.load(fp)
        data_root = Path(config['DATA_DIR'])
        output_dir = Path(config['OUTPUT_DIR'])

    model_path = output_dir / args.architecture / f'net_{args.epoch:04d}.params'
    settings_path = data_root / 'settings.pkl'
    with open(str(settings_path), 'rb') as f:
      settings = pickle.load(f)
    imsizes = [(settings['imsize'][0] // (2 ** s), settings['imsize'][1] // (2 ** s)) for s in range(4)]
    K = settings['K']
    Ki = np.linalg.inv(K)
    baseline = settings['baseline']
    pat = settings['pattern']

    d2d = networks.DispToDepth(focal_length= K[0, 0],baseline= baseline).cuda()
    lcn_in = networks.LCN(5, 0.05).cuda()

    pat = pat.mean(axis=2)
    pat = torch.from_numpy(pat[None][None].astype(np.float32)).to('cuda')
    pat_lcn, _ = lcn_in(pat)
    pat_cat = torch.cat((pat_lcn, pat), dim=1)

    if args.architecture == 'single_frame':
        net = networks.DispDecoder(channels_in=2, max_disp=128, imsizes=imsizes).cuda().eval()
    elif args.architecture == 'multi_frame':
        net = multi_frame_networks.FuseNet(imsize=imsizes[0], K=K, baseline=baseline, max_disp=128).cuda().eval()

    net.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        sample_list = [os.path.join(data_root, o) for o in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,o))]
        for sample_path in tqdm(sample_list, ascii = True):
            with h5py.File(os.path.join(sample_path, f'frames.hdf5'), "r") as frames_sample:
                if args.architecture == 'single_frame':
                    im = torch.tensor(frames_sample['im']).cuda()
                    im_lcn, im_std = lcn_in(im)
                    im = torch.cat([im_lcn, im], dim=1)
                    disp = net(im)[0].cpu().numpy()
                elif args.architecture == 'multi_frame':
                    im = torch.tensor(frames_sample['im']).cuda()
                    im_lcn, im_std = lcn_in(im)
                    im = torch.cat([im_lcn, im], dim=1)

                    amb = torch.tensor(frames_sample['ambient']).cuda()

                    R = torch.tensor(frames_sample['R']).cuda()
                    t = torch.tensor(frames_sample['t']).cuda()

                    flow = {}
                    with h5py.File(os.path.join(sample_path, f'flow.hdf5'), "r") as sample_flow:
                        for i0 in range(4):
                            for i1 in range(4):
                                if i0 != i1:
                                    flow[f'flow_{i0}{i1}'] = torch.tensor(sample_flow[f'flow_{i0}{i1}'][:]).cuda()

                    with h5py.File(os.path.join(sample_path, f'single_frame_disp.hdf5'), "r") as sample_primary_disp:
                        primary_disp = torch.tensor(sample_primary_disp['disp'][:]).cuda()

                    disp = net(im.unsqueeze(1), amb.unsqueeze(1), primary_disp.unsqueeze(1), d2d(primary_disp.unsqueeze(1)), R.unsqueeze(1), t.unsqueeze(1), flow).detach().cpu().numpy()
                    disp = disp[:, 0]

            with h5py.File(os.path.join(sample_path, f'{args.architecture}_disp.hdf5'), "w") as f:
                f.create_dataset('disp', data=disp)