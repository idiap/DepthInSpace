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

import h5py
import numpy as np
import pickle
import cv2
import os

from . import base_dataset
from .data_manipulation import augment_image


class TrackSynDataset(base_dataset.BaseDataset):
  '''
  Load locally saved synthetic dataset
  '''
  def __init__(self, settings_path, sample_paths, track_length=2, train=True, data_aug=False, load_flow_data = False, load_primary_data = False, load_pseudo_gt = False, data_type = 'synthetic'):
    super().__init__(train=train)

    self.settings_path = settings_path
    self.sample_paths = sample_paths
    self.data_aug = data_aug
    self.train = train
    self.track_length=track_length
    self.load_flow_data = load_flow_data
    self.load_primary_data = load_primary_data
    self.load_pseudo_gt = load_pseudo_gt
    self.data_type = data_type
    assert(track_length<=4)

    with open(str(settings_path), 'rb') as f:
      settings = pickle.load(f)
    self.imsizes = [(settings['imsize'][0] // (2 ** s), settings['imsize'][1] // (2 ** s)) for s in range(4)]
    self.patterns = []
    for imsize in self.imsizes:
      pat = cv2.resize(settings['pattern'], (imsize[1], imsize[0]), interpolation=cv2.INTER_LINEAR)
      self.patterns.append(pat)
    self.baseline = settings['baseline']
    self.K = settings['K']
    self.focal_lengths = [self.K[0,0]/(2**s) for s in range(4)]

    self.scale = len(self.imsizes)

    self.max_shift = 0
    self.max_blur = 0.5
    self.max_noise = 3.0
    self.max_sp_noise = 0.0005

  def __len__(self):
    return len(self.sample_paths)

  def __getitem__(self, idx):
    if not self.train:
      rng = self.get_rng(idx)
    else:
      rng = np.random.RandomState()
    sample_path = self.sample_paths[idx]

    if self.train:
      track_ind = np.random.permutation(4)[0:self.track_length]
    else:
      track_ind = np.arange(0, self.track_length)

    ret = {}
    ret['id'] = idx

    with h5py.File(os.path.join(sample_path,f'frames.hdf5'), "r") as sample_frames:
      # load imgs, at all scales
      for sidx in range(len(self.imsizes)):
        im = sample_frames['im']
        amb = sample_frames['ambient']
        grad = sample_frames['grad']
        if sidx == 0:
          ret[f'im{sidx}'] = np.stack([im[tidx, ...] for tidx in track_ind], axis=0)
          ret[f'ambient{sidx}'] = np.stack([amb[tidx, ...] for tidx in track_ind], axis=0)
          ret[f'grad{sidx}'] = np.stack([grad[tidx, ...] for tidx in track_ind], axis=0)
        else:
          ret[f'im{sidx}'] = np.stack([cv2.resize(im[tidx, 0, ...], self.imsizes[sidx][::-1])[None] for tidx in track_ind], axis=0)
          ret[f'ambient{sidx}'] = np.stack([cv2.resize(amb[tidx, 0, ...], self.imsizes[sidx][::-1])[None] for tidx in track_ind], axis=0)
          ret[f'grad{sidx}'] = np.stack([cv2.resize(grad[tidx, 0, ...], self.imsizes[sidx][::-1])[None] for tidx in track_ind], axis=0)

      # load disp and grad only at full resolution
      ret[f'disp0'] = np.stack([sample_frames['disp'][tidx, ...] for tidx in track_ind], axis=0)
      ret['R'] = np.stack([sample_frames['R'][tidx, ...] for tidx in track_ind], axis=0)
      ret['t'] = np.stack([sample_frames['t'][tidx, ...] for tidx in track_ind], axis=0)
      if self.data_type == 'real':
        ret[f'sgm_disp'] = np.stack([sample_frames['sgm_disp'][tidx, ...] for tidx in track_ind], axis=0)

    if self.load_flow_data:
      with h5py.File(os.path.join(sample_path, f'flow.hdf5'), "r") as sample_flow:
        for i0, tidx0 in enumerate(track_ind):
          for i1, tidx1 in enumerate(track_ind):
            if tidx0 != tidx1:
              ret[f'flow_{i0}{i1}'] = sample_flow[f'flow_{tidx0}{tidx1}'][:]

    if self.load_primary_data:
      with h5py.File(os.path.join(sample_path, f'single_frame_disp.hdf5'), "r") as sample_primary_disp:
        ret[f'primary_disp'] = np.stack([sample_primary_disp['disp'][tidx, ...] for tidx in track_ind], axis=0)

    if self.load_pseudo_gt:
      with h5py.File(os.path.join(sample_path, f'multi_frame_disp.hdf5'), "r") as sample_disp:
        ret[f'pseudo_gt'] = np.stack([sample_disp['disp'][tidx, ...] for tidx in track_ind], axis=0)

    #### apply data augmentation at different scales seperately, only work for max_shift=0
    if self.data_aug:
      for sidx in range(len(self.imsizes)):
        if sidx==0:
          img = ret[f'im{sidx}']
          amb = ret[f'ambient{sidx}']
          disp = ret[f'disp{sidx}']
          if self.load_primary_data:
            primary_disp = ret[f'primary_disp']
          else:
            primary_disp = None
          if self.data_type == 'real':
            sgm_disp = ret[f'sgm_disp']
          else:
            sgm_disp = None
          grad = ret[f'grad{sidx}']
          img_aug = np.zeros_like(img)
          amb_aug = np.zeros_like(img)
          disp_aug = np.zeros_like(img)
          primary_disp_aug = np.zeros_like(img)
          sgm_disp_aug = np.zeros_like(img)
          grad_aug = np.zeros_like(img)
          for i in range(img.shape[0]):
            if self.load_primary_data:
              primary_disp_i = primary_disp[i,0]
            else:
              primary_disp_i = None
            if self.data_type == 'real':
              sgm_disp_i = sgm_disp[i,0]
            else:
              sgm_disp_i = None
            img_aug_, amb_aug_, disp_aug_, primary_disp_aug_, sgm_disp_aug_, grad_aug_ = augment_image(img[i,0],rng,
                                                           amb=amb[i,0],disp=disp[i,0],primary_disp=primary_disp_i, sgm_disp= sgm_disp_i,grad=grad[i,0],
                                                           max_shift=self.max_shift, max_blur=self.max_blur,
                                                           max_noise=self.max_noise, max_sp_noise=self.max_sp_noise)
            img_aug[i] = img_aug_[None].astype(np.float32)
            amb_aug[i] = amb_aug_[None].astype(np.float32)
            disp_aug[i] = disp_aug_[None].astype(np.float32)
            if self.load_primary_data:
              primary_disp_aug[i] = primary_disp_aug_[None].astype(np.float32)
            if self.data_type == 'real':
              sgm_disp_aug[i] = sgm_disp_aug_[None].astype(np.float32)
            grad_aug[i] = grad_aug_[None].astype(np.float32)
          ret[f'im{sidx}'] = img_aug
          ret[f'ambient{sidx}'] = amb_aug
          ret[f'disp{sidx}'] = disp_aug
          if self.load_primary_data:
            ret[f'primary_disp'] = primary_disp_aug
          if self.data_type == 'real':
            ret[f'sgm_disp'] = sgm_disp_aug
          ret[f'grad{sidx}'] = grad_aug
        else:
          img = ret[f'im{sidx}']
          img_aug = np.zeros_like(img)
          for i in range(img.shape[0]):
            img_aug_, _, _, _, _, _ = augment_image(img[i,0],rng,
                                           max_shift=self.max_shift, max_blur=self.max_blur,
                                           max_noise=self.max_noise, max_sp_noise=self.max_sp_noise)
            img_aug[i] = img_aug_[None].astype(np.float32)
          ret[f'im{sidx}'] = img_aug

    return ret

  def getK(self, sidx=0):
    K = self.K.copy() / (2**sidx)
    K[2,2] = 1
    return K

        

if __name__ == '__main__':
  pass

