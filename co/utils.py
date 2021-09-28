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
import time
from collections import OrderedDict
import argparse
import subprocess

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class StopWatch(object):
  def __init__(self):
    self.timings = OrderedDict()
    self.starts = {}

  def start(self, name):
    self.starts[name] = time.time()

  def stop(self, name):
    if name not in self.timings:
      self.timings[name] = []
    self.timings[name].append(time.time() - self.starts[name])

  def get(self, name=None, reduce=np.sum):
    if name is not None:
      return reduce(self.timings[name])
    else:
      ret = {}
      for k in self.timings:
        ret[k] = reduce(self.timings[k])
      return ret

  def __repr__(self):
    return ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get().items()])
  def __str__(self):
    return ', '.join(['%s: %f[s]' % (k,v) for k,v in self.get().items()])

class ETA(object):
  def __init__(self, length):
    self.length = length
    self.start_time = time.time()
    self.current_idx = 0
    self.current_time = time.time()

  def update(self, idx):
    self.current_idx = idx
    self.current_time = time.time()

  def get_elapsed_time(self):
    return self.current_time - self.start_time

  def get_item_time(self):
    return self.get_elapsed_time() / (self.current_idx + 1)

  def get_remaining_time(self):
    return self.get_item_time() * (self.length - self.current_idx + 1)

  def format_time(self, seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    hours = int(hours)
    minutes = int(minutes)
    return f'{hours:02d}:{minutes:02d}:{seconds:05.2f}'

  def get_elapsed_time_str(self):
    return self.format_time(self.get_elapsed_time())

  def get_remaining_time_str(self):
    return self.format_time(self.get_remaining_time())

def git_hash(cwd=None):
  ret = subprocess.run(['git', 'describe', '--always'], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  hash = ret.stdout
  if hash is not None and 'fatal' not in hash.decode():
    return hash.decode().strip()
  else:
    return None

