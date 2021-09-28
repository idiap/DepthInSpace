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
import numpy as np
import logging
import itertools
import matplotlib.pyplot as plt
import co

from data import base_dataset
from data import dataset

from model import networks

from . import worker

class Worker(worker.Worker):
  def __init__(self, args, **kwargs):
    super().__init__(args, **kwargs)

    self.disparity_loss = networks.DisparitySmoothLoss()

  def get_train_set(self):
    train_set = dataset.TrackSynDataset(self.settings_path, self.train_paths, train=True, data_aug=True, track_length=self.track_length, load_flow_data = True, load_primary_data = False, load_pseudo_gt = self.use_pseudo_gt, data_type = self.data_type)
    return train_set

  def get_test_sets(self):
    test_sets = base_dataset.TestSets()
    test_set = dataset.TrackSynDataset(self.settings_path, self.test_paths, train=False, data_aug=False, track_length=self.track_length, load_flow_data = True, load_primary_data = False, load_pseudo_gt = self.use_pseudo_gt, data_type = self.data_type)
    test_sets.append('simple', test_set, test_frequency=1)

    self.patterns = []
    self.ph_losses = []
    self.ge_losses = []
    self.d2ds = []

    self.lcn_in = self.lcn_in.to('cuda')
    for sidx in range(len(test_set.imsizes)):
      imsize = test_set.imsizes[sidx]
      pat = test_set.patterns[sidx]
      pat = pat.mean(axis=2)
      pat = torch.from_numpy(pat[None][None].astype(np.float32)).to('cuda')
      pat,_ = self.lcn_in(pat)

      self.patterns.append(pat)

      pat = torch.cat([pat for idx in range(3)], dim=1)
      ph_loss = networks.RectifiedPatternSimilarityLoss(imsize[0],imsize[1], pattern=pat)

      K = test_set.getK(sidx)
      Ki = np.linalg.inv(K)
      K = torch.from_numpy(K)
      Ki = torch.from_numpy(Ki)
      ge_loss = networks.Single_Frame_Flow_Consistency_Loss(K, Ki, imsize[0], imsize[1], clamp=0.1)

      self.ph_losses.append( ph_loss )
      self.ge_losses.append( ge_loss )

      d2d = networks.DispToDepth(float(test_set.focal_lengths[sidx]), float(test_set.baseline))
      self.d2ds.append( d2d )

    return test_sets

  def net_forward(self, net, flow = None):
    im0 = self.data['im0']
    tl = im0.shape[0]
    bs = im0.shape[1]
    im0 = im0.view(-1, *im0.shape[2:])
    out = net(im0)

    if not(isinstance(out, tuple) or isinstance(out, list)):
        out = out.view(tl, bs, *out.shape[1:])
    else:
        out = [o.view(tl, bs, *o.shape[1:]) for o in out]

    return out

  def loss_forward(self, out, train, flow_out = None):
    if not(isinstance(out, tuple) or isinstance(out, list)):
      out = [out]

    vals = []

    # apply photometric loss
    for s,o in zip(itertools.count(), out):
      im = self.data[f'im0']
      im = im.view(-1, *im.shape[2:])
      o = o.view(-1, *o.shape[2:])
      std = self.data[f'std0']
      std = std.view(-1, *std.shape[2:])
      val, pattern_proj = self.ph_losses[0](o, im[:,0:1,...], std)
      vals.append(val / (2 ** s))

    # apply disparity loss
    for s, o in zip(itertools.count(), out):
      if s == 0:
        amb0 = self.data[f'ambient0']
        amb0 = amb0.contiguous().view(-1, *amb0.shape[2:])
        o = o.view(-1, *o.shape[2:])
        val = self.disparity_loss(o, amb0)
        vals.append(val * 0.4 / (2 ** s))

    # apply geometric loss
    R = self.data['R']
    t = self.data['t']
    amb = self.data['ambient0']
    ge_num = self.track_length * (self.track_length-1) / 2
    for sidx in range(1):
      d2d = self.d2ds[0]
      depth = d2d(out[sidx])
      ge_loss = self.ge_losses[0]
      for tidx0 in range(depth.shape[0]):
        for tidx1 in range(tidx0+1, depth.shape[0]):
          depth0 = depth[tidx0]
          R0 = R[tidx0]
          t0 = t[tidx0]
          amb0 = amb[tidx0]
          flow0 = flow_out[f'flow_{tidx0}{tidx1}']
          depth1 = depth[tidx1]
          R1 = R[tidx1]
          t1 = t[tidx1]
          amb1 = amb[tidx1]
          flow1 = flow_out[f'flow_{tidx1}{tidx0}']

          val, flow_mask0, flow_mask1, orig_mask = ge_loss(depth0, depth1, R0, t0, R1, t1, flow0, flow1, amb0, amb1)
          vals.append(val * 0.2 / ge_num / (2 ** sidx))

    # using pseudo-ground truth
    if self.use_pseudo_gt:
      for s, o in zip(itertools.count(), out):
        val = torch.mean(torch.abs(o - self.data['pseudo_gt']))
        vals.append(val * 0.1 / (2 ** s))

    # warming up the network for a few epochs
    if train and self.data_type == 'real':
      if self.current_epoch < self.warmup_epochs:
        for s, o in zip(itertools.count(), out):
          valid_mask = (self.data['sgm_disp'] > 30).float()
          val = torch.sum(torch.abs(o - self.data['sgm_disp'] + 1.5 * torch.randn(o.size()).cuda()) * valid_mask) / torch.sum(valid_mask)
          vals.append(val * 0.1)

    return vals

  def numpy_in_out(self, output):
    if not(isinstance(output, tuple) or isinstance(output, list)):
      output = [output]
    es = output[0].detach().to('cpu').numpy()
    gt = self.data['disp0'].detach().to('cpu').numpy().astype(np.float32)
    im = self.data['im0'][:,:,0:1,...].detach().to('cpu').numpy()
    amb = self.data['ambient0'].detach().to('cpu').numpy()
    pat = self.patterns[0].detach().to('cpu').numpy()

    es = es * (gt > 0)

    return es, gt, im, amb, pat

  def write_img(self, out_path, es, gt, im, amb, pat):
    logging.info(f'write img {out_path}')

    diff = np.abs(es - gt)

    vmin, vmax = np.nanmin(gt), np.nanmax(gt)
    vmin = vmin - 0.2*(vmax-vmin)
    vmax = vmax + 0.2*(vmax-vmin)

    vmax = np.max([vmax, 16])

    fig = plt.figure(figsize=(16,16))
    es0 = co.cmap.color_depth_map(es[0], scale=vmax)
    gt0 = co.cmap.color_depth_map(gt[0], scale=vmax)
    diff0 = co.cmap.color_error_image(diff[0], BGR=True)

    # plot pattern and input images
    ax = plt.subplot(3,3,1); plt.imshow(pat, vmin=pat.min(), vmax=pat.max(), cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'Projector Pattern')
    ax = plt.subplot(3,3,2); plt.imshow(im[0], vmin=im.min(), vmax=im.max(), cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 IR Input')
    ax = plt.subplot(3,3,3); plt.imshow(amb[0], vmin=amb.min(), vmax=amb.max(), cmap='gray'); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 Ambient Input')

    # plot disparities, ground truth disparity is shown only for reference
    ax = plt.subplot(3,3,4); plt.imshow(gt0[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 Disparity GT {np.nanmin(gt[0]):.4f}/{np.nanmax(gt[0]):.4f}')
    ax = plt.subplot(3,3,5); plt.imshow(es0[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 Disparity Est. {es[0].min():.4f}/{es[0].max():.4f}')
    ax = plt.subplot(3,3,6); plt.imshow(diff0[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F0 Disparity Err. {diff[0].mean():.5f}')

    es1 = co.cmap.color_depth_map(es[1], scale=vmax)
    gt1 = co.cmap.color_depth_map(gt[1], scale=vmax)
    diff1 = co.cmap.color_error_image(diff[1], BGR=True)
    ax = plt.subplot(3,3,7); plt.imshow(gt1[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F1 Disparity GT {np.nanmin(gt[1]):.4f}/{np.nanmax(gt[1]):.4f}')
    ax = plt.subplot(3,3,8); plt.imshow(es1[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F1 Disparity Est. {es[1].min():.4f}/{es[1].max():.4f}')
    ax = plt.subplot(3,3,9); plt.imshow(diff1[...,[2,1,0]]); plt.xticks([]); plt.yticks([]); ax.set_title(f'F1 Disparity Err. {diff[1].mean():.5f}')

    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close(fig)

  def callback_train_post_backward(self, net, errs, output, epoch, batch_idx, masks):
    if batch_idx % 256 == 0:
      out_path = self.exp_output_dir / f'train_{epoch:03d}_{batch_idx:04d}.png'
      es, gt, im, amb, pat = self.numpy_in_out(output)
      self.write_img(out_path, es[:, 0, 0], gt[:, 0, 0], im[:, 0, 0], amb[:, 0, 0], pat[0, 0])
      torch.cuda.empty_cache()

  def callback_test_start(self, epoch, set_idx):
    self.metric = co.metric.MultipleMetric(
        co.metric.DistanceMetric(vec_length=1),
        co.metric.OutlierFractionMetric(vec_length=1, thresholds=[0.1, 0.5, 1, 2, 5]) 
      )

  def callback_test_add(self, epoch, set_idx, batch_idx, n_batches, output, masks):
    es, gt, im, amb, pat = self.numpy_in_out(output)

    if batch_idx % 8 == 0:
      out_path = self.exp_output_dir / f'test_{epoch:03d}_{batch_idx:04d}.png'
      self.write_img(out_path, es[:, 0, 0], gt[:, 0, 0], im[:, 0, 0], amb[:, 0, 0], pat[0, 0])

    es = self.crop_reshape(es)
    gt = self.crop_reshape(gt)
    self.metric.add(es, gt)

  def crop_reshape(self, input):
    output = input.reshape(-1, 1)
    return output

  def callback_test_stop(self, epoch, set_idx, loss):
    logging.info(f'{self.metric}')
    for k, v in self.metric.items():
      self.metric_add_test(epoch, set_idx, k, v)

if __name__ == '__main__':
  pass
