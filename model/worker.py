# DepthInSpace is a PyTorch-based program which estimates 3D depth maps
# from active structured-light sensor's multiple video frames.
#
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
#
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

import pickle

import numpy as np
import torch
import os
import random
import logging
import datetime
from pathlib import Path
import argparse
import socket
import gc
import json
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
from model import networks


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

class Worker(object):
  def __init__(self, args, seed=42, test_batch_size=4, num_workers=4, save_frequency=1, train_device='cuda:0', test_device='cuda:0', max_train_iter=-1):
    self.use_pseudo_gt = args.use_pseudo_gt
    self.lcn_radius = args.lcn_radius
    self.track_length = args.track_length
    self.data_type = args.data_type
    # assert(self.track_length>1)

    self.architecture = args.architecture
    self.epochs = args.epochs
    self.warmup_epochs = args.warmup_epochs
    self.seed = seed
    self.train_batch_size = args.train_batch_size
    self.test_batch_size = test_batch_size
    self.num_workers = num_workers
    self.save_frequency = save_frequency
    self.train_device = train_device
    self.test_device = test_device
    self.max_train_iter = max_train_iter

    self.errs_list=[]

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    with open(config_path) as fp:
      config = json.load(fp)
      data_root = Path(config['DATA_DIR'])
      output_dir = Path(config['OUTPUT_DIR'])
    self.settings_path = data_root / 'settings.pkl'
    self.output_dir = output_dir
    with open(str(self.settings_path), 'rb') as f:
      settings = pickle.load(f)
      self.baseline = settings['baseline']
      self.K = settings['K']
      self.Ki = np.linalg.inv(self.K)
      self.imsizes = [settings['imsize']]
      for iter in range(3):
        self.imsizes.append((int(self.imsizes[-1][0] / 2), int(self.imsizes[-1][1] / 2)))
    self.ref_pattern = settings['pattern']

    sample_paths = sorted((data_root).glob('0*/'))
    if self.data_type == 'synthetic':
      self.train_paths = sample_paths[2**10:]
      self.test_paths = sample_paths[2**9:2**10]
      self.valid_paths = sample_paths[0:2**9]
    elif self.data_type == 'real':
      self.test_paths = sample_paths[4::8]
      self.train_paths = [path for path in sample_paths if path not in self.test_paths]

    self.lcn_in = networks.LCN(self.lcn_radius, 0.05).cuda()

    self.setup_experiment()

  def setup_experiment(self):
    self.exp_output_dir = self.output_dir / self.architecture
    self.exp_output_dir.mkdir(parents=True, exist_ok=True)

    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler( str(self.exp_output_dir / 'train.log') ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )

    logging.info('='*80)
    logging.info(f'Start of experiment with architecture: {self.architecture}')
    logging.info(socket.gethostname())
    self.log_datetime()
    logging.info('='*80)

    self.metric_path = self.exp_output_dir / 'metrics.json'
    if self.metric_path.exists():
      with open(str(self.metric_path), 'r') as fp:
        self.metric_data = json.load(fp)
    else:
      self.metric_data = {}

    self.init_seed()

  def metric_add_train(self, epoch, key, val):
    epoch = str(epoch)
    key = str(key)
    if epoch not in self.metric_data:
      self.metric_data[epoch] = {}
    if 'train' not in self.metric_data[epoch]:
      self.metric_data[epoch]['train'] = {}
    self.metric_data[epoch]['train'][key] = val

  def metric_add_test(self, epoch, set_idx, key, val):
    epoch = str(epoch)
    set_idx = str(set_idx)
    key = str(key)
    if epoch not in self.metric_data:
      self.metric_data[epoch] = {}
    if 'test' not in self.metric_data[epoch]:
      self.metric_data[epoch]['test'] = {}
    if set_idx not in self.metric_data[epoch]['test']:
      self.metric_data[epoch]['test'][set_idx] = {}
    self.metric_data[epoch]['test'][set_idx][key] = val

  def metric_save(self):
    with open(str(self.metric_path), 'w') as fp:
      json.dump(self.metric_data, fp, indent=2)

  def init_seed(self, seed=None):
    if seed is not None:
      self.seed = seed
    logging.info(f'Set seed to {self.seed}')
    np.random.seed(self.seed)
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed(self.seed)

  def log_datetime(self):
    logging.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

  def mem_report(self):
    for obj in gc.get_objects():
      if torch.is_tensor(obj):
          print(type(obj), obj.shape)

  def get_net_path(self, epoch, root=None):
    if root is None:
      root = self.exp_output_dir
    return root / f'net_{epoch:04d}.params'

  def get_do_parser_cmds(self):
    return ['retrain', 'resume', 'retest', 'test_init']

  def get_do_parser(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', type=str, default='resume', choices=self.get_do_parser_cmds())
    parser.add_argument('--epoch', type=int, default=-1)
    return parser

  def do_cmd(self, args, net, optimizer, scheduler=None):
    if args.cmd == 'retrain':
      self.train(net, optimizer, resume=False, scheduler=scheduler)
    elif args.cmd == 'resume':
      self.train(net, optimizer, resume=True, scheduler=scheduler)
    elif args.cmd == 'retest':
      self.retest(net, epoch=args.epoch)
    elif args.cmd == 'test_init':
      test_sets = self.get_test_sets()
      self.test(-1, net, test_sets)
    else:
      raise Exception('invalid cmd')

  def do(self, net, optimizer, load_net_optimizer=None, scheduler=None):
    parser = self.get_do_parser()
    args, _ = parser.parse_known_args()

    if load_net_optimizer is not None and args.cmd not in ['schedule']:
      net, optimizer = load_net_optimizer()

    self.do_cmd(args, net, optimizer, scheduler=scheduler)

  def retest(self, net, epoch=-1):
    if epoch < 0:
      epochs = range(self.epochs)
    else:
      epochs = [epoch]

    test_sets = self.get_test_sets()

    for epoch in epochs:
      net_path = self.get_net_path(epoch)
      if net_path.exists():
        state_dict = torch.load(str(net_path))
        net.load_state_dict(state_dict)
        self.test(epoch, net, test_sets)

  def format_err_str(self, errs, div=1):
    err = sum(errs)
    if len(errs) > 1:
      err_str = f'{err/div:0.4f}=' + '+'.join([f'{e/div:0.4f}' for e in errs])
    else:
      err_str = f'{err/div:0.4f}'
    return err_str

  def write_err_img(self):
    err_img_path = self.exp_output_dir / 'errs.png'
    fig = plt.figure(figsize=(16,16))
    lines=[]
    for idx,errs in enumerate(self.errs_list):
      line,=plt.plot(range(len(errs)), errs, label=f'error{idx}')
      lines.append(line)
    plt.tight_layout()
    plt.legend(handles=lines)
    plt.savefig(str(err_img_path))
    plt.close(fig)


  def callback_train_new_epoch(self, epoch, net, optimizer):
    pass

  def train(self, net, optimizer, resume=False, scheduler=None):
    logging.info('='*80)
    logging.info('Start training')
    self.log_datetime()
    logging.info('='*80)

    train_set = self.get_train_set()
    test_sets = self.get_test_sets()

    net = net.to(self.train_device)

    epoch = 0
    min_err = {ts.name: 1e9 for ts in test_sets}

    state_path = self.exp_output_dir / 'state.dict'
    if resume and state_path.exists():
      logging.info('='*80)
      logging.info(f'Loading state from {state_path}')
      logging.info('='*80)
      state = torch.load(str(state_path))
      epoch = state['epoch'] + 1
      if 'min_err' in state:
        min_err = state['min_err']

      curr_state = net.state_dict()
      curr_state.update(state['state_dict'])
      net.load_state_dict(curr_state)

      try:
        optimizer.load_state_dict(state['optimizer'])
      except:
        logging.info('Warning: cannot load optimizer from state_dict')
        pass
      if 'cpu_rng_state' in state:
        torch.set_rng_state(state['cpu_rng_state'])
      if 'gpu_rng_state' in state:
        torch.cuda.set_rng_state(state['gpu_rng_state'])

    for epoch in range(epoch, self.epochs):
      self.current_epoch = epoch
      self.callback_train_new_epoch(epoch, net, optimizer)

      # train epoch
      self.train_epoch(epoch, net, optimizer, train_set)

      # test epoch
      errs = self.test(epoch, net, test_sets)

      if (epoch + 1) % self.save_frequency == 0:
        net = net.to(self.train_device)

        state_dict = {
            'epoch': epoch,
            'min_err': min_err,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state(),
        }
        logging.info(f'save state to {state_path}')
        state_path = self.exp_output_dir / 'state.dict'
        torch.save(state_dict, str(state_path))

        for test_set_name in errs:
          err = sum(errs[test_set_name])
          if err < min_err[test_set_name]:
            min_err[test_set_name] = err
            state_path = self.exp_output_dir / f'state_set_{test_set_name}_best.dict'
            logging.info(f'save state to {state_path}')
            torch.save(state_dict, str(state_path))

        # store network
        net_path = self.get_net_path(epoch)
        logging.info(f'save network to {net_path}')
        torch.save(net.state_dict(), str(net_path))

      if scheduler is not None:
        scheduler.step()

    logging.info('='*80)
    logging.info('Finished training')
    self.log_datetime()
    logging.info('='*80)

  def get_train_set(self):
    raise NotImplementedError()

  def get_test_sets(self):
    raise NotImplementedError()

  def copy_data(self, data, device, requires_grad, train):
    self.data = {}

    self.lcn_in = self.lcn_in.to(device)
    for key, val in data.items():
      # from
      # batch_size x track_length x ...
      # to
      # track_length x batch_size x ...
      if len(val.shape)>2:
        val = val.transpose(0, 1)
      self.data[key] = val.to(device)
      if 'im' in key and 'blend' not in key and 'primary' not in key:
        im = self.data[key]
        tl = im.shape[0]
        bs = im.shape[1]
        im_lcn,im_std = self.lcn_in(im.contiguous().view(-1, *im.shape[2:]))
        key_std = key.replace('im','std')
        self.data[key_std] = im_std.view(tl, bs, *im.shape[2:]).to(device)
        im_cat = torch.cat((im_lcn.view(tl, bs, *im.shape[2:]), im), dim=2)
        self.data[key] = im_cat
      elif key == 'ambient0':
        ambient = self.data[key]
        tl = ambient.shape[0]
        bs = ambient.shape[1]
        ambient_lcn, ambient_std = self.lcn_in(ambient.contiguous().view(-1, *ambient.shape[2:]))
        ambient_cat = torch.cat((ambient_lcn.view(tl, bs, *ambient.shape[2:]), ambient), dim=2)
        self.data[f'{key}_in'] = ambient_cat.to(device).requires_grad_(requires_grad=requires_grad)

    # Mimicing the reference pattern
    pat = self.ref_pattern.mean(axis=2)
    pat = torch.from_numpy(pat[None][None].astype(np.float32)).to('cuda')
    pat_lcn, _ = self.lcn_in(pat)
    pat_cat = torch.cat((pat_lcn, pat), dim=1).unsqueeze(0)
    self.data[f'ref_pattern'] = pat_cat.repeat([*self.data['im0'].shape[0:2], 1, 1, 1])

  def net_forward(self, net, train):
    raise NotImplementedError()

  def read_optical_flow(self, train):
    im = self.data['ambient0']
    out = {}
    for tidx0 in range(im.shape[0]):
      for tidx1 in range(im.shape[0]):
        if tidx0 != tidx1:
          out[f'flow_{tidx0}{tidx1}'] = self.data[f'flow_{tidx0}{tidx1}'][0]

    return out

  def loss_forward(self, output, train, flow_out):
    raise NotImplementedError()

  def callback_train_post_backward(self, net, errs, output, epoch, batch_idx, masks):
    pass

  def callback_train_start(self, epoch):
    pass

  def callback_train_stop(self, epoch, loss):
    pass

  def train_epoch(self, epoch, net, optimizer, dset):
    self.callback_train_start(epoch)
    stopwatch = StopWatch()

    logging.info('='*80)
    logging.info('Train epoch %d' % epoch)

    dset.current_epoch = epoch
    train_loader = torch.utils.data.DataLoader(dset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=False)

    net = net.to(self.train_device)
    net.train()

    mean_loss = None

    n_batches = self.max_train_iter if self.max_train_iter > 0 else len(train_loader)
    bar = ETA(length=n_batches)

    stopwatch.start('total')
    stopwatch.start('data')
    for batch_idx, data in enumerate(train_loader):
      if self.max_train_iter > 0 and batch_idx > self.max_train_iter: break
      self.copy_data(data, device=self.train_device, requires_grad=False, train=True)
      stopwatch.stop('data')

      optimizer.zero_grad()

      stopwatch.start('forward')
      flow_output = self.read_optical_flow(train=True)
      output = self.net_forward(net, flow_output)

      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('forward')

      stopwatch.start('loss')
      errs = self.loss_forward(output, True, flow_output)
      if isinstance(errs, dict):
        masks = errs['masks']
        errs = errs['errs']
      else:
        masks = []
      if not isinstance(errs, list) and not isinstance(errs, tuple):
        errs = [errs]
      err = sum(errs)
      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('loss')

      stopwatch.start('backward')
      err.backward()
      self.callback_train_post_backward(net, errs, output, epoch, batch_idx, masks)
      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('backward')

      # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
      # print('Max Allocated:', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')
      # print('Max Cached:', round(torch.cuda.max_memory_cached(0) / 1024 ** 3, 1), 'GB')

      stopwatch.start('optimizer')
      optimizer.step()
      if 'cuda' in self.train_device: torch.cuda.synchronize()
      stopwatch.stop('optimizer')

      bar.update(batch_idx)
      if (epoch <= 1 and batch_idx < 128) or batch_idx % 16 == 0:
        err_str = self.format_err_str(errs)
        logging.info(f'train e{epoch}: {batch_idx+1}/{len(train_loader)}: loss={err_str} | {bar.get_elapsed_time_str()} / {bar.get_remaining_time_str()}')
        #self.write_err_img()


      if mean_loss is None:
        mean_loss = [0 for e in errs]
      for erridx, err in enumerate(errs):
        mean_loss[erridx] += err.item()

      stopwatch.start('data')
    stopwatch.stop('total')
    logging.info('timings: %s' % stopwatch)

    mean_loss = [l / len(train_loader) for l in mean_loss]
    self.callback_train_stop(epoch, mean_loss)
    self.metric_add_train(epoch, 'loss', mean_loss)

    # save metrics
    self.metric_save()

    err_str = self.format_err_str(mean_loss)
    logging.info(f'avg train_loss={err_str}')
    return mean_loss

  def callback_test_start(self, epoch, set_idx):
    pass

  def callback_test_add(self, epoch, set_idx, batch_idx, n_batches, output, masks):
    pass

  def callback_test_stop(self, epoch, set_idx, loss):
    pass

  def test(self, epoch, net, test_sets):
    errs = {}
    for test_set_idx, test_set in enumerate(test_sets):
      if (epoch + 1) % test_set.test_frequency == 0:
        logging.info('='*80)
        logging.info(f'testing set {test_set.name}')
        err = self.test_epoch(epoch, test_set_idx, net, test_set.dset)
        errs[test_set.name] = err
    return errs

  def test_epoch(self, epoch, set_idx, net, dset):
    logging.info('-'*80)
    logging.info('Test epoch %d' % epoch)
    dset.current_epoch = epoch
    test_loader = torch.utils.data.DataLoader(dset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, pin_memory=False)

    net = net.to(self.test_device)
    net.eval()

    with torch.no_grad():
      mean_loss = None

      self.callback_test_start(epoch, set_idx)

      bar = ETA(length=len(test_loader))
      stopwatch = StopWatch()
      stopwatch.start('total')
      stopwatch.start('data')
      for batch_idx, data in enumerate(test_loader):
        self.copy_data(data, device=self.test_device, requires_grad=False, train=False)
        stopwatch.stop('data')

        stopwatch.start('forward')
        flow_output = self.read_optical_flow(train=False)

        output = self.net_forward(net, flow_output)

        if 'cuda' in self.test_device: torch.cuda.synchronize()
        stopwatch.stop('forward')

        stopwatch.start('loss')
        errs = self.loss_forward(output, False, flow_output)
        if isinstance(errs, dict):
          masks = errs['masks']
          errs = errs['errs']
        else:
          masks = []
        if not isinstance(errs, list) and not isinstance(errs, tuple):
          errs = [errs]

        bar.update(batch_idx)
        if batch_idx % 25 == 0:
          err_str = self.format_err_str(errs)
          logging.info(f'test e{epoch}: {batch_idx+1}/{len(test_loader)}: loss={err_str} | {bar.get_elapsed_time_str()} / {bar.get_remaining_time_str()}')

        if mean_loss is None:
          mean_loss = [0 for e in errs]
        for erridx, err in enumerate(errs):
          mean_loss[erridx] += err.item()
        stopwatch.stop('loss')

        self.callback_test_add(epoch, set_idx, batch_idx, len(test_loader), output, masks)

        stopwatch.start('data')
      stopwatch.stop('total')
      logging.info('timings: %s' % stopwatch)

      mean_loss = [l / len(test_loader) for l in mean_loss]
      self.callback_test_stop(epoch, set_idx, mean_loss)
      self.metric_add_test(epoch, set_idx, 'loss', mean_loss)

      # save metrics
      self.metric_save()

      err_str = self.format_err_str(mean_loss)
      logging.info(f'test epoch {epoch}: avg test_loss={err_str}')
      return mean_loss
