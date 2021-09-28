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

import numpy as np

from . import utils

class StopWatch(utils.StopWatch):
  def __del__(self):
    print('='*80)
    print('gtimer:')
    total = ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get(reduce=np.sum).items()])
    print(f'  [total]  {total}')
    mean = ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get(reduce=np.mean).items()])
    print(f'  [mean]   {mean}')
    median = ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get(reduce=np.median).items()])
    print(f'  [median] {median}')
    print('='*80)

GTIMER = StopWatch()

def start(name):
  GTIMER.start(name)
def stop(name):
  GTIMER.stop(name)

class Ctx(object):
  def __init__(self, name):
    self.name = name

  def __enter__(self):
    start(self.name)

  def __exit__(self, *args):
    stop(self.name)
