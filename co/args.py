# DepthInSpace is a PyTorch-based program which estimates 3D depth maps
# from active structured-light sensor's multiple video frames.
#
# MIT License
#
# Copyright (c) 2019 autonomousvision
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

import argparse
from .utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--data_type',
                        default='synthetic', choices=['synthetic', 'real'], type=str)
    #
    parser.add_argument('--cmd',
                        help='Start training or test', 
                        default='resume', choices=['retrain', 'resume', 'retest', 'test_init'], type=str)
    parser.add_argument('--epoch',
                        help='If larger than -1, retest on the specified epoch',
                        default=-1, type=int)
    parser.add_argument('--epochs',
                        help='Training epochs',
                        default=100, type=int)
    parser.add_argument('--warmup_epochs',
                        help='Number of epochs where SGM Disparities are used as supervisor when training on the real dataset',
                        default=150, type=int)
    #
    parser.add_argument('--lcn_radius',
                        help='Radius of the window for LCN pre-processing',
                        default=5, type=int)
    parser.add_argument('--max_disp',
                        help='Maximum disparity',
                        default=128, type=int)
    #
    parser.add_argument('--track_length',
                        help='Track length for geometric loss',
                        default=4, type=int)
    #
    parser.add_argument('--train_batch_size',
                        help='Train Batch Size',
                        default=8, type=int)
    #
    parser.add_argument('--architecture',
                        help='The architecture which will be used',
                        default='single_frame', choices=['single_frame', 'multi_frame'], type=str)
    #
    parser.add_argument('--use_pseudo_gt',
                        help='Only applicable in single-frame model',
                        default=False, type=str2bool)

    args = parser.parse_args()

    return args