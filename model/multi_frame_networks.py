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

import cv2

import torch
import numpy as np

from torch.utils.checkpoint import checkpoint

from .networks import TimedModule, OutputLayerFactory


def merge_tl_bs(x):
  return x.contiguous().view(-1, *x.shape[2:])

def split_tl_bs(x, tl, bs):
  return x.contiguous().view(tl, bs, *x.shape[1:])

def resize_like(x, target):
  x_shape = x.shape[:-3]
  height = target.shape[-2]
  width = target.shape[-1]

  out = torch.nn.functional.interpolate(x.contiguous().view(-1, *x.shape[-3:]), size=(height, width), mode='bilinear',
                                        align_corners=True)
  out = out.view(*x_shape, *out.shape[1:])

  return out


def resize_flow_like(flow, target):
  height = target.shape[-2]
  width = target.shape[-1]

  out = {}
  for key, val in flow.items():
    flow_height = val.shape[-2]
    flow_width = val.shape[-1]

    resized_flow = torch.nn.functional.interpolate(val, size=(height, width), mode='bilinear', align_corners=True)
    resized_flow[:, 0, :, :] *= float(width) / float(flow_width)
    resized_flow[:, 1, :, :] *= float(height) / float(flow_height)
    out[key] = resized_flow

  return out

def resize_flow_masks_like(flow_masks, target):
  height = target.shape[-2]
  width = target.shape[-1]

  with torch.no_grad():
    out = {}
    for key, val in flow_masks.items():
      resized_mask = torch.nn.functional.interpolate(val, size=(height, width), mode='bilinear', align_corners=True)

      out[key] = (resized_mask > 0.5).float()

  return out

def warp(x, flow):
  width = x.shape[-1]
  height = x.shape[-2]

  u, v = np.meshgrid(range(width), range(height))
  u = torch.from_numpy(u.astype('float32')).to(x.device)
  v = torch.from_numpy(v.astype('float32')).to(x.device)

  uv_prj = flow.clone().permute(0, 2, 3, 1)
  uv_prj[..., 0] += u
  uv_prj[..., 1] += v

  uv_prj[..., 0] = 2 * (uv_prj[..., 0] / (width - 1) - 0.5)
  uv_prj[..., 1] = 2 * (uv_prj[..., 1] / (height - 1) - 0.5)
  x_prj = torch.nn.functional.grid_sample(x, uv_prj, padding_mode='zeros', align_corners=True)

  return x_prj

class FuseNet(TimedModule):
  '''
  Fuse Net
  '''
  def __init__(self, imsize, K, baseline, track_length = 4, block_num = 4, channels = 32, max_disp= 128, movement_mask_en = 1):
    super(FuseNet, self).__init__(mod_name='FuseNet')
    self.movement_mask_en = movement_mask_en

    self.im_height = imsize[0]
    self.im_width = imsize[1]
    self.core_height = self.im_height // 2
    self.core_width = self.im_width // 2
    self.K = K
    self.Ki = np.linalg.inv(K)
    self.baseline = baseline
    self.track_length = track_length
    self.block_num = block_num
    self.channels = channels
    self.max_disp = max_disp

    u, v = np.meshgrid(range(self.im_width), range(self.im_height))
    u = cv2.resize(u, (self.core_width, self.core_height), interpolation = cv2.INTER_NEAREST)
    v = cv2.resize(v, (self.core_width, self.core_height), interpolation = cv2.INTER_NEAREST)
    uv = np.stack((u,v,np.ones_like(u)), axis=2).reshape(-1,3)

    ray = uv @ self.Ki.T
    ray = ray.reshape(1, 1,-1,3).astype(np.float32)
    self.ray = torch.from_numpy(ray).cuda()

    self.conv1 = self.conv(4, self.channels // 2, kernel_size=4, stride=2)
    self.conv2 = self.conv(self.channels // 2, self.channels // 2, kernel_size=3, stride=1)

    # self.conv3 = self.conv(self.channels // 2, self.channels, kernel_size=4, stride=2)
    self.conv3 = self.conv(self.channels // 2, self.channels, kernel_size=3, stride=1)
    self.conv4 = self.conv(self.channels, self.channels, kernel_size=3, stride=1)

    self.res1 = ResNetBlock(self.channels)
    self.res2 = ResNetBlock(self.channels)
    self.res3 = ResNetBlock(self.channels)

    self.blocks = torch.nn.ModuleList([Block2D3D(channels = self.channels, tl= self.track_length) for i in range(self.block_num)])

    self.upconv1 = self.upconv(self.channels, self.channels)
    self.upconv2 = self.upconv(self.channels, self.channels)

    self.amb_conv = self.conv(1, 16, kernel_size=3, stride=1)
    self.amb_res1 = ResNetBlock(16)
    self.amb_res2 = ResNetBlock(16)

    self.ref_conv = self.conv(16 + self.channels, 32, kernel_size=3, stride=1)
    self.ref_res1 = ResNetBlock(32)
    self.ref_res2 = ResNetBlock(32)
    self.ref_res3 = ResNetBlock(32)

    self.final_conv = self.conv(32, 16, kernel_size=3, stride=1)

    self.predict_disp = OutputLayerFactory( type='disp', params={ 'alpha': self.max_disp, 'beta': 0, 'gamma': 1, 'offset': 3})(16)

  def conv(self, in_planes, out_planes, kernel_size=3, stride=1):
    return torch.nn.Sequential(
      torch.nn.ZeroPad2d((kernel_size - 1) // 2),
      torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0),
      torch.nn.SELU(inplace=True)
    )

  def upconv(self, in_planes, out_planes):
    return torch.nn.Sequential(
      torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
      torch.nn.SELU(inplace=True)
    )

  def unproject(self, d, R, t):
    tl = d.shape[0]
    bs = d.shape[1]
    xyz = d.view(tl, bs, -1, 1) * self.ray
    xyz = xyz - t.view(tl, bs, 1, 3)
    xyz = torch.matmul(xyz, R)

    return xyz

  def change_view_angle(self, xyz, R, t):
    xyz_changed = torch.matmul(xyz, R.transpose(1,2))
    xyz_changed = xyz_changed + t.unsqueeze(1).unsqueeze(0)

    return xyz_changed

  def gather_warped_xyz(self, tidx, xyz, depth, flow, amb):
    tl = xyz.shape[0]
    bs = xyz.shape[1]

    frame_inds = [i for i in range(tl) if i != tidx]

    warped_xyz = []
    warped_xyz.append(xyz[tidx].transpose(1, 2).view(bs, 3, self.core_height, self.core_width))

    warped_mask = []
    warped_mask.append(torch.ones(bs, 1, self.core_height, self.core_width).to(xyz.device))

    for j in frame_inds:
      warped_xyz.append(warp(xyz[j].transpose(1, 2).view(bs, 3, self.core_height, self.core_width), flow[f'flow_{tidx}{j}']))

      with torch.no_grad():
        flow0 = flow[f'flow_{tidx}{j}'].detach()
        flow10 = warp(flow[f'flow_{j}{tidx}'].detach(), flow0)
        fb_mask = ((flow0.detach() + flow10) ** 2).sum(dim=1) < 0.5 + 0.01 * (
                  (flow0.detach() ** 2).sum(dim=1) + (flow10 ** 2).sum(dim=1))
        fb_mask = fb_mask.type(torch.float32).unsqueeze(1)

        warped_mask.append(fb_mask)

    warped_xyz = torch.stack(warped_xyz, dim=0)
    warped_mask = torch.stack(warped_mask, dim=0)

    return warped_xyz, warped_mask

  def pre_process(self, input_data, d, checkpoint_var):
    out_conv1 = self.conv1(torch.cat([input_data, d], dim= 1))
    out_conv2 = self.conv2(out_conv1)

    out_conv3 = self.conv3(out_conv2)
    out_conv4 = self.conv4(out_conv3)

    out_res1 = self.res1(out_conv4)
    out_res2 = self.res2(out_res1)
    feat = self.res3(out_res2)

    return feat

  def process_amb(self, amb, feat):
    out_amb_conv = self.amb_conv(merge_tl_bs(amb))
    out_amb_res1 = self.amb_res1(out_amb_conv)
    out_amb_res2 = self.amb_res2(out_amb_res1)

    out_process_upc = self.process_upc(feat, out_amb_res2)

    return out_process_upc

  def process_upc(self, feat, out_process_amb):
    # out_upconv1 = self.upconv1(feat)
    # out_upconv2 = self.upconv2(out_upconv1)
    # out_upconv = out_upconv2

    # out_upconv1 = self.upconv1(feat)
    # out_upconv = out_upconv1

    out_upconv = torch.nn.functional.interpolate(feat, size=(self.im_height, self.im_width), mode='bilinear',
                                                 align_corners=True)

    out_ref_conv = self.ref_conv(torch.cat([out_upconv, out_process_amb], dim=1))

    return out_ref_conv

  def post_process(self, feat, amb):
    out_process_amb = checkpoint(self.process_amb, amb, feat, preserve_rng_state= False)
    # out_process_amb = self.process_amb(amb, feat)

    out_ref_res1 = checkpoint(self.ref_res1, out_process_amb, preserve_rng_state= False)
    # out_ref_res1 = self.ref_res1(out_process_amb)
    out_ref_res2 = checkpoint(self.ref_res2, out_ref_res1, preserve_rng_state= False)
    # out_ref_res2 = self.ref_res2(out_ref_res1)
    out_ref_res3 = checkpoint(self.ref_res3, out_ref_res2, preserve_rng_state= False)
    # out_ref_res3 = self.ref_res3(out_ref_res2)

    out_final_conv = self.final_conv(out_ref_res3)
    disp = self.predict_disp(out_final_conv)

    return disp

  def tforward(self, ir, amb, d, depth, R, t, flow):
    tl = ir.shape[0]
    bs = ir.shape[1]

    input_data = merge_tl_bs(torch.cat((ir, amb), 2))
    checkpoint_var = torch.tensor([0.0]).to(input_data.device).requires_grad_(True)
    # feat = checkpoint(self.pre_process, input_data, merge_tl_bs(d), checkpoint_var, preserve_rng_state= False)
    feat = self.pre_process(input_data, merge_tl_bs(d), checkpoint_var)

    ###### Block Part
    core_feat = split_tl_bs(feat, tl, bs)
    core_depth = resize_like(depth, core_feat)
    core_flow = resize_flow_like(flow, core_feat)
    core_amb = resize_like(amb, core_feat)
    xyz = self.unproject(core_depth, R, t)

    warped_xyz = []
    warped_mask = []
    for tidx in range(tl):
      xyz_changed = self.change_view_angle(xyz, R[tidx], t[tidx])
      w_xyz, w_mask = self.gather_warped_xyz(tidx, xyz_changed, core_depth, core_flow, core_amb)

      warped_xyz.append(w_xyz)
      warped_mask.append(w_mask)
    warped_xyz = torch.stack(warped_xyz, dim=0)
    warped_mask = torch.stack(warped_mask, dim=0)

    for block in self.blocks:
      core_feat = block(core_feat, warped_xyz, warped_mask, core_flow)

    feat = merge_tl_bs(core_feat)
    #### End of Block Part

    disp = self.post_process(feat, amb)
    out = disp.view(tl, bs, *disp.shape[1:])

    return out

class Block2D3D(TimedModule):
  def __init__(self, channels, tl):
    super(Block2D3D, self).__init__(mod_name='Block2D3D')

    self.channels = channels
    self.tl = tl

    self.conv_mf = self.conv(self.channels * self.tl, self.channels, kernel_size=1, stride=1, activation='none')

    self.conv1_1 = self.conv(self.channels, self.channels, kernel_size=3, stride=1, activation='relu')
    self.conv1_2 = self.conv(self.channels, self.channels, kernel_size=3, stride=1, activation='relu')

    self.conv2_1 = self.conv(self.channels, self.channels, kernel_size=4, stride=2, activation='relu')
    self.conv2_2 = self.conv(self.channels, self.channels, kernel_size=3, stride=1, activation='relu')

    self.conv_fuse = self.conv(self.channels * 3, self.channels, kernel_size=3, stride=1, activation='none')

    # self.conv_res = self.conv(self.channels, self.channels, kernel_size=3, stride=1, activation='relu')
    self.activation_res = torch.nn.SELU(inplace=True)

    self.conv3d_1 = Conv3D(channels_in= self.channels, channels_out= self.channels, tl= self.tl, stride= 2)
    self.conv3d_2 = Conv3D(channels_in= self.channels, channels_out= self.channels, tl= self.tl, stride= 1)

  def conv(self, in_planes, out_planes, kernel_size=3, stride=1, activation = 'none'):
    if activation == 'none':
      return torch.nn.Sequential(
        torch.nn.ZeroPad2d((kernel_size - 1) // 2),
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0),
        # torch.nn.BatchNorm2d(out_planes)
        torch.nn.GroupNorm(num_groups= 1, num_channels= out_planes)
      )
    elif activation == 'relu':
      return torch.nn.Sequential(
        torch.nn.ZeroPad2d((kernel_size - 1) // 2),
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0),
        torch.nn.SELU(inplace=True),
        # torch.nn.BatchNorm2d(out_planes),
        torch.nn.GroupNorm(num_groups=1, num_channels=out_planes)
      )

  def gather_warped_feat(self, tidx, feat, flow):
    tl = feat.shape[0]

    frame_inds = [i for i in range(tl) if i != tidx]

    warped_feat = []
    warped_feat.append(feat[tidx])

    for j in frame_inds:
      warped_feat.append(warp(feat[j], flow[f'flow_{tidx}{j}']))

    warped_feat = torch.stack(warped_feat, dim=0)

    return warped_feat

  def tforward(self, feat, warped_xyz, warped_mask, flow):
    self.flow = flow

    out_conv3d_1, warped_feat = checkpoint(self.fwd_3d_1, feat, warped_xyz, warped_mask, preserve_rng_state= False)
    # out_conv3d_1, warped_feat = self.fwd_3d_1(feat, warped_xyz, warped_mask)

    out_conv3d_2, _ = checkpoint(self.fwd_3d_2, out_conv3d_1, warped_xyz, warped_mask, preserve_rng_state= False)
    # out_conv3d_2, _ = self.fwd_3d_2(out_conv3d_1, warped_xyz, warped_mask)

    out = checkpoint(self.fwd_2d, feat, warped_feat, warped_mask, out_conv3d_2, preserve_rng_state= False)
    # out = self.fwd_2d(feat, warped_feat, warped_mask, out_conv3d_2)

    return out

  def fwd_3d_1(self, feat, warped_xyz, warped_mask):
    tl = feat.shape[0]

    warped_feat = []
    out_conv3d = []
    for tidx in range(tl):
      warped_feat.append(self.gather_warped_feat(tidx, feat, self.flow))
      out_conv3d.append(self.conv3d_1(warped_xyz[tidx], warped_feat[-1], warped_mask[tidx]))
    warped_feat = torch.stack(warped_feat, dim=0)
    out_conv3d = torch.stack(out_conv3d, dim=0)

    return out_conv3d, warped_feat

  def fwd_3d_2(self, feat, warped_xyz, warped_mask):
    tl = feat.shape[0]

    resized_flow = resize_flow_like(self.flow, feat)
    resized_warped_xyz = resize_like(warped_xyz, feat)
    resized_warped_mask = (resize_like(warped_mask, feat) > 0.5).float()

    warped_feat = []
    out_conv3d = []
    for tidx in range(tl):
      warped_feat.append(self.gather_warped_feat(tidx, feat, resized_flow))
      out_conv3d.append(self.conv3d_2(resized_warped_xyz[tidx], warped_feat[-1], resized_warped_mask[tidx]))
    warped_feat = torch.stack(warped_feat, dim=0)
    out_conv3d = torch.stack(out_conv3d, dim=0)

    return out_conv3d, warped_feat

  def fwd_2d(self, feat, warped_feat, warped_mask, out_conv3d_2):
    tl = feat.shape[0]
    bs = feat.shape[1]

    warped_feat_2d = (warped_feat * warped_mask / warped_mask.mean(dim=1, keepdim=True)).transpose(1, 2)
    warped_feat_2d = warped_feat_2d.reshape(tl * bs, -1, *warped_feat_2d.shape[4:])

    out_conv_mf = self.conv_mf(warped_feat_2d)

    out_conv1_1 = self.conv1_1(out_conv_mf)
    out_conv1_2 = self.conv1_2(out_conv1_1)

    out_conv2_1 = self.conv2_1(out_conv_mf)
    out_conv2_2 = self.conv2_2(out_conv2_1)
    out_ups2 = torch.nn.functional.interpolate(out_conv2_2, scale_factor=2, mode='bilinear', align_corners=True)

    out_ups3d = torch.nn.functional.interpolate(merge_tl_bs(out_conv3d_2), scale_factor=2, mode='bilinear', align_corners=True)

    ### Fusion Part
    out_fuse = torch.cat((out_conv1_2, out_ups2, out_ups3d), dim= 1)
    out_conv_fuse = self.conv_fuse(out_fuse)

    out = self.activation_res(split_tl_bs(out_conv_fuse, tl, bs) + feat)

    return out

class Conv3D(TimedModule):
  def __init__(self, channels_in, channels_out, neighbors = 9, tl = 4, ksize = 3, stride = 1, radius_sq = 0.04):
    super(Conv3D, self).__init__(mod_name='Conv3D')
    self.channels_in = channels_in
    self.channels_out = channels_out
    self.neighbors = neighbors
    self.tl = tl
    self.ksize = ksize
    self.stride = stride
    self.radius_sq = radius_sq

    self.dense1 = self.dense(3, self.channels_out // 2, activation= 'selu')
    self.dense2 = self.dense(self.channels_out // 2, self.channels_out, activation= 'selu')

    self.w = torch.nn.Parameter(torch.zeros([self.channels_out, self.channels_out]))
    torch.nn.init.xavier_uniform_(self.w, gain=0.1)

    self.activation = torch.nn.SELU(inplace=True)
    # self.bn = torch.nn.BatchNorm2d(channels_out)
    self.bn = torch.nn.GroupNorm(num_groups= 1, num_channels= channels_out)

  def dense(self, in_planes, out_planes, activation = 'selu'):
    if activation == 'selu':
      return torch.nn.Sequential(
        torch.nn.Linear(in_planes, out_planes),
        torch.nn.SELU(inplace=True),
      )
    elif activation == 'softmax':
      return torch.nn.Sequential(
        torch.nn.Linear(in_planes, out_planes),
        torch.nn.Softmax(dim = -1)
      )
    elif activation == 'none':
      return torch.nn.Sequential(
        torch.nn.Linear(in_planes, out_planes),
      )

  def tforward(self, xyz, feat, mask, checkpoint_var = 0):
    padding_len = (self.ksize - 1) // 2

    xyz = torch.nn.functional.pad(xyz, (padding_len, padding_len, padding_len, padding_len), mode='constant', value=0)
    feat = torch.nn.functional.pad(feat, (padding_len, padding_len, padding_len, padding_len), mode='constant', value=0)
    mask = torch.nn.functional.pad(mask, (padding_len, padding_len, padding_len, padding_len), mode='constant', value=0)

    xyz = xyz.unfold(3, self.ksize, self.stride).unfold(4, self.ksize, self.stride)
    feat = feat.unfold(3, self.ksize, self.stride).unfold(4, self.ksize, self.stride)
    mask = mask.unfold(3, self.ksize, self.stride).unfold(4, self.ksize, self.stride)

    xyz = xyz.permute(1, 3, 4, 5, 6, 0, 2)      # (bs, h, w, k, k, tl, c)
    feat = feat.permute(1, 3, 4, 5, 6, 0, 2)
    mask = mask.permute(1, 3, 4, 5, 6, 0, 2)

    bs_h_w = xyz.shape[0:3]
    xyz = xyz.reshape(-1, self.ksize * self.ksize * self.tl, xyz.shape[-1])   # (?, k*k*tl, c)
    feat = feat.reshape(-1, self.ksize * self.ksize * self.tl, feat.shape[-1])
    mask = mask.reshape(-1, self.ksize * self.ksize * self.tl, mask.shape[-1])

    xyz_plane = xyz / (xyz[..., 2:] + 1e-12)

    tidx = ((self.ksize ** 2) // 2) * self.tl
    xyz_local = xyz - xyz[:, tidx:tidx + 1, :]
    xyz_plane_local = xyz_plane - xyz_plane[:, tidx:tidx + 1, :]

    xyz_sq = (xyz_plane_local ** 2).sum(dim=-1, keepdim=True)

    xyz_max_copy = (mask * xyz_sq) + (1 - mask) * (xyz_sq.max() + 1)
    _, neighbors_ind = torch.topk(xyz_max_copy, self.neighbors, dim= 1, largest=False, sorted=False)
    xyz_neighbors = torch.gather(xyz_local, dim = 1, index= neighbors_ind.expand(-1, -1, xyz_local.shape[-1]))
    feat_neighbors = torch.gather(feat, dim = 1, index= neighbors_ind.expand(-1, -1, feat.shape[-1]))

    out_dense1 = self.dense1(xyz_neighbors)
    out_dense2 = self.dense2(out_dense1)

    feat_weighted = (out_dense2 * feat_neighbors).sum(dim = 1)

    out_conv = torch.matmul(feat_weighted, self.w).view(*bs_h_w, self.channels_out).permute(0, 3, 1, 2)
    out_conv = self.activation(out_conv)

    out = self.bn(out_conv)

    return out

class ResNetBlock(TimedModule):
  def __init__(self, planes):
    super(ResNetBlock, self).__init__(mod_name='ResNetBlock')

    self.pad = torch.nn.ZeroPad2d(1)

    self.conv1 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0)
    # self.bn1 = torch.nn.BatchNorm2d(planes)
    self.bn1 = torch.nn.GroupNorm(num_groups= 1, num_channels= planes)
    self.relu1 = torch.nn.SELU(inplace=True)
    self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0)
    # self.bn2 = torch.nn.BatchNorm2d(planes)
    self.bn2 = torch.nn.GroupNorm(num_groups= 1, num_channels= planes)
    self.relu2 = torch.nn.SELU(inplace=True)

  def forward(self, x):
    identity = x.clone()

    out = self.conv1(self.pad(x))
    out = self.relu1(out)
    out = self.bn1(out)

    out = self.conv2(self.pad(out))
    out = self.bn2(out)

    out += identity
    out = self.relu2(out)

    return out