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

import torch
import torch.nn.functional as F
import numpy as np

import co

from . import ext_functions

class TimedModule(torch.nn.Module):
  def __init__(self, mod_name):
    super().__init__()
    self.mod_name = mod_name

  def tforward(self, *args, **kwargs):
    raise Exception('not implemented')

  def forward(self, *args, **kwargs):
    torch.cuda.synchronize()
    with co.gtimer.Ctx(self.mod_name):
      x = self.tforward(*args, **kwargs)
      torch.cuda.synchronize()
    return x


class PosOutput(TimedModule):
  def __init__(self, channels_in, type, im_height, im_width, alpha=1, beta=0, gamma=1, offset=0):
    super().__init__(mod_name='PosOutput')
    self.im_width = im_width
    self.im_width = im_width

    if type == 'pos':
      self.layer = torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, 1, kernel_size=3, padding=1),
        SigmoidAffine(alpha=alpha, beta=beta, gamma=gamma, offset=offset)
      )
    elif type == 'pos_row':
      self.layer = torch.nn.Sequential(
          MultiLinear(im_height, channels_in, 1),
          SigmoidAffine(alpha=alpha, beta=beta, gamma=gamma, offset=offset)
        )

    self.u_pos = None

  def tforward(self, x):
    if self.u_pos is None:
      self.u_pos = torch.arange(x.shape[3], dtype=torch.float32).view(1,1,1,-1)
    self.u_pos = self.u_pos.to(x.device)
    pos = self.layer(x)
    disp = self.u_pos - pos
    return disp


class OutputLayerFactory(object):
  '''
  Define type of output
  type options:
    linear: apply only conv channel, used for the edge decoder
    disp: estimate the disparity
    disp_row: independently estimate the disparity per row 
    pos: estimate the absolute location 
    pos_row: independently estimate the absolute location per row 
  '''
  def __init__(self, type='disp', params={}):
    self.type = type
    self.params = params

  def __call__(self, channels_in, imsize = None):

    if self.type == 'linear':
       return torch.nn.Conv2d(channels_in, 1, kernel_size=3, padding=1)

    elif self.type == 'disp':
      return torch.nn.Sequential(
          torch.nn.Conv2d(channels_in, 1, kernel_size=3, padding=1),
          SigmoidAffine(**self.params)
        )

    elif self.type == 'disp_row':
      return torch.nn.Sequential(
          MultiLinear(imsize[0], channels_in, 1),
          SigmoidAffine(**self.params)
        )

    elif self.type == 'pos' or self.type == 'pos_row':
      return PosOutput(channels_in, **self.params)

    else:
      raise Exception('unknown output layer type')


class SigmoidAffine(TimedModule):
  def __init__(self, alpha=1, beta=0, gamma=1, offset=0):
    super().__init__(mod_name='SigmoidAffine')
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.offset = offset

  def tforward(self, x):
    return torch.sigmoid(x/self.gamma - self.offset) * self.alpha + self.beta


class MultiLinear(TimedModule):
  def __init__(self, n, channels_in, channels_out):
    super().__init__(mod_name='MultiLinear')
    self.channels_out = channels_out
    self.mods = torch.nn.ModuleList()
    for idx in range(n):
      self.mods.append(torch.nn.Linear(channels_in, channels_out))

  def tforward(self, x):
    x = x.permute(2,0,3,1) # BxCxHxW => HxBxWxC
    y = x.new_empty(*x.shape[:-1], self.channels_out)
    for hidx in range(x.shape[0]):
      y[hidx] = self.mods[hidx](x[hidx])
    y = y.permute(1,3,0,2) # HxBxWxC => BxCxHxW
    return y



class DispNetS(TimedModule):
  '''
  Disparity Decoder based on DispNetS
  '''
  def __init__(self, channels_in, imsizes, output_facs, coordconv=False, weight_init=False, channel_multiplier=1):
    super(DispNetS, self).__init__(mod_name='DispNetS')

    conv_planes = channel_multiplier * np.array( [32, 64, 128, 256, 512, 512, 512] )
    self.conv1 = self.downsample_conv(channels_in, conv_planes[0], kernel_size=7)
    self.conv2 = self.downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
    self.conv3 = self.downsample_conv(conv_planes[1], conv_planes[2])
    self.conv4 = self.downsample_conv(conv_planes[2], conv_planes[3])
    self.conv5 = self.downsample_conv(conv_planes[3], conv_planes[4])
    self.conv6 = self.downsample_conv(conv_planes[4], conv_planes[5])
    self.conv7 = self.downsample_conv(conv_planes[5], conv_planes[6])

    upconv_planes = channel_multiplier * np.array( [512, 512, 256, 128, 64, 32, 16] )
    self.upconv7 = self.upconv(conv_planes[6],   upconv_planes[0])
    self.upconv6 = self.upconv(upconv_planes[0], upconv_planes[1])
    self.upconv5 = self.upconv(upconv_planes[1], upconv_planes[2])
    self.upconv4 = self.upconv(upconv_planes[2], upconv_planes[3])
    self.upconv3 = self.upconv(upconv_planes[3], upconv_planes[4])
    self.upconv2 = self.upconv(upconv_planes[4], upconv_planes[5])
    self.upconv1 = self.upconv(upconv_planes[5], upconv_planes[6])

    self.iconv7 = self.conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
    self.iconv6 = self.conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
    self.iconv5 = self.conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
    self.iconv4 = self.conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
    self.iconv3 = self.conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
    self.iconv2 = self.conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
    self.iconv1 = self.conv(1 + upconv_planes[6], upconv_planes[6])


    if isinstance(output_facs, list):
      self.predict_disp4 = output_facs[3](upconv_planes[3], imsizes[3])
      self.predict_disp3 = output_facs[2](upconv_planes[4], imsizes[2])
      self.predict_disp2 = output_facs[1](upconv_planes[5], imsizes[1])
      self.predict_disp1 = output_facs[0](upconv_planes[6], imsizes[0])
    else:
      self.predict_disp4 = output_facs(upconv_planes[3], imsizes[3])
      self.predict_disp3 = output_facs(upconv_planes[4], imsizes[2])
      self.predict_disp2 = output_facs(upconv_planes[5], imsizes[1])
      self.predict_disp1 = output_facs(upconv_planes[6], imsizes[0])

  # def init_weights(self):
  #   for m in self.modules():
  #     if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
  #       torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
  #       if m.bias is not None:
  #         torch.nn.init.zeros_(m.bias)

  def downsample_conv(self, in_planes, out_planes, kernel_size=3):
    return torch.nn.Sequential(
      torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
      torch.nn.ReLU(inplace=True)
    )

  def conv(self, in_planes, out_planes):
    return torch.nn.Sequential(
      torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
      torch.nn.ReLU(inplace=True)
    )

  def upconv(self, in_planes, out_planes):
    return torch.nn.Sequential(
      torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
      torch.nn.ReLU(inplace=True)
    )

  def crop_like(self, input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

  def tforward(self, x):
    out_conv1 = self.conv1(x)
    out_conv2 = self.conv2(out_conv1)
    out_conv3 = self.conv3(out_conv2)
    out_conv4 = self.conv4(out_conv3)
    out_conv5 = self.conv5(out_conv4)
    out_conv6 = self.conv6(out_conv5)
    out_conv7 = self.conv7(out_conv6)

    out_upconv7 = self.crop_like(self.upconv7(out_conv7), out_conv6)
    concat7 = torch.cat((out_upconv7, out_conv6), 1)
    out_iconv7 = self.iconv7(concat7)

    out_upconv6 = self.crop_like(self.upconv6(out_iconv7), out_conv5)
    concat6 = torch.cat((out_upconv6, out_conv5), 1)
    out_iconv6 = self.iconv6(concat6)

    out_upconv5 = self.crop_like(self.upconv5(out_iconv6), out_conv4)
    concat5 = torch.cat((out_upconv5, out_conv4), 1)
    out_iconv5 = self.iconv5(concat5)

    out_upconv4 = self.crop_like(self.upconv4(out_iconv5), out_conv3)
    concat4 = torch.cat((out_upconv4, out_conv3), 1)
    out_iconv4 = self.iconv4(concat4)
    disp4 = self.predict_disp4(out_iconv4)

    out_upconv3 = self.crop_like(self.upconv3(out_iconv4), out_conv2)
    disp4_up = self.crop_like(torch.nn.functional.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
    concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
    out_iconv3 = self.iconv3(concat3)
    disp3 = self.predict_disp3(out_iconv3)

    out_upconv2 = self.crop_like(self.upconv2(out_iconv3), out_conv1)
    disp3_up = self.crop_like(torch.nn.functional.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
    concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
    out_iconv2 = self.iconv2(concat2)
    disp2 = self.predict_disp2(out_iconv2)

    out_upconv1 = self.crop_like(self.upconv1(out_iconv2), x)
    disp2_up = self.crop_like(torch.nn.functional.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
    concat1 = torch.cat((out_upconv1, disp2_up), 1)
    out_iconv1 = self.iconv1(concat1)
    disp1 = self.predict_disp1(out_iconv1)

    out1 = disp1
    out2 = torch.nn.functional.interpolate(input=disp2, size=(out1.size(2), out1.size(3)), mode='bilinear', align_corners=False)
    out3 = torch.nn.functional.interpolate(input=disp3, size=(out1.size(2), out1.size(3)), mode='bilinear', align_corners=False)
    out4 = torch.nn.functional.interpolate(input=disp4, size=(out1.size(2), out1.size(3)), mode='bilinear', align_corners=False)

    return (out1, out2, out3, out4)

class DispDecoder(TimedModule):
  '''
  Disparity Decoder
  '''
  def __init__(self, *args, max_disp=128, **kwargs):
    super(DispDecoder, self).__init__(mod_name='DispDecoder')

    output_facs_disp = [OutputLayerFactory( type='disp', params={ 'alpha': max_disp/(2**s), 'beta': 0, 'gamma': 1, 'offset': 3}) for s in range(4)]
    self.disp_decoder = DispNetS(*args, output_facs=output_facs_disp, **kwargs)

  def tforward(self, x):
    disp = self.disp_decoder(x)
    return disp

class DispToDepth(TimedModule):
  def __init__(self, focal_length, baseline):
    super().__init__(mod_name='DispToDepth')
    self.baseline_focal_length = baseline * focal_length

  def tforward(self, disp):
    disp = torch.nn.functional.relu(disp) + 1e-12
    depth = self.baseline_focal_length / disp
    return depth

class PosToDepth(DispToDepth):
  def __init__(self, focal_length, baseline, im_height, im_width):
    super().__init__(focal_length, baseline)
    self.mod_name = 'PosToDepth'

    self.im_height = im_height
    self.im_width = im_width
    self.u_pos = torch.arange(im_width, dtype=torch.float32).view(1,1,1,-1)

  def tforward(self, pos):
    self.u_pos = self.u_pos.to(pos.device)
    disp = self.u_pos - pos
    return super().forward(disp)


class RectifiedPatternSimilarityLoss(TimedModule):
  '''
  Photometric Loss
  '''
  def __init__(self, im_height, im_width, pattern, loss_type='census_sad', loss_eps=0.5):
    super().__init__(mod_name='RectifiedPatternSimilarityLoss')
    self.im_height = im_height
    self.im_width = im_width
    self.pattern = pattern.mean(dim=1, keepdim=True).contiguous()

    u, v = np.meshgrid(range(im_width), range(im_height))
    uv0 = np.stack((u,v), axis=2).reshape(-1,1)
    uv0 = uv0.astype(np.float32).reshape(1,-1,2)
    self.uv0 = torch.from_numpy(uv0)

    self.loss_type = loss_type
    self.loss_eps = loss_eps

  def tforward(self, disp0, im, std=None, output_mean=True):
    self.pattern = self.pattern.to(disp0.device)
    self.uv0 = self.uv0.to(disp0.device)

    uv0 = self.uv0.expand(disp0.shape[0], *self.uv0.shape[1:])
    uv1 = torch.empty_like(uv0)
    uv1[...,0] = uv0[...,0] - disp0.contiguous().view(disp0.shape[0],-1)
    uv1[...,1] = uv0[...,1]

    uv1[..., 0] = 2 * (uv1[..., 0] / (self.im_width-1) - 0.5)
    uv1[..., 1] = 2 * (uv1[..., 1] / (self.im_height-1) - 0.5)
    uv1 = uv1.view(-1, self.im_height, self.im_width, 2).clone()
    pattern = self.pattern.expand(disp0.shape[0], *self.pattern.shape[1:])
    pattern_proj = torch.nn.functional.grid_sample(pattern, uv1, padding_mode='border', align_corners=True)
    mask = torch.ones_like(im)
    if std is not None:
      mask = mask*std

    diff = ext_functions.photometric_loss(pattern_proj.contiguous(), im.contiguous(), 9, self.loss_type, self.loss_eps)
    if output_mean:
      val = (mask*diff).sum() / mask.sum()
    else:
      val = diff
    return val, pattern_proj

class SSIM(TimedModule):
  """Layer to compute the SSIM loss between a pair of images
  """
  def __init__(self):
    super().__init__(mod_name='SSIM')
    self.mu_x_pool = torch.nn.AvgPool2d(3, 1)
    self.mu_y_pool = torch.nn.AvgPool2d(3, 1)
    self.sig_x_pool = torch.nn.AvgPool2d(3, 1)
    self.sig_y_pool = torch.nn.AvgPool2d(3, 1)
    self.sig_xy_pool = torch.nn.AvgPool2d(3, 1)

    self.refl = torch.nn.ReflectionPad2d(1)

    self.C1 = 0.01 ** 2
    self.C2 = 0.03 ** 2

  def tforward(self, x, y):
    x = self.refl(x)
    y = self.refl(y)

    mu_x = self.mu_x_pool(x)
    mu_y = self.mu_y_pool(y)

    sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
    sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
    sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class DisparitySmoothLoss(TimedModule):
  '''
  Depth Smooth Loss
  '''
  def __init__(self):
    super().__init__(mod_name='DepthSmoothLoss')
    self.sobel = SobelFilter(norm=False, ksize=5)

  def tforward(self, disp, im):
    self.sobel=self.sobel.to(disp.device)

    mean_disp = disp.mean(2, True).mean(3, True)
    # norm_disp = disp / (mean_disp + 1e-7)
    norm_disp = disp

    grad = self.sobel(norm_disp)
    grad_im = self.sobel(im)

    val = torch.abs(grad * torch.exp(-torch.abs(255 * grad_im)))

    return val.mean()

class ProjectionBaseLoss(TimedModule):
  '''
  Base module of the Geometric Loss
  '''
  def __init__(self, K, Ki, im_height, im_width):
    super().__init__(mod_name='ProjectionBaseLoss')

    self.K = K.view(-1,3,3)

    self.im_height = im_height
    self.im_width = im_width

    u, v = np.meshgrid(range(im_width), range(im_height))
    uv = np.stack((u,v,np.ones_like(u)), axis=2).reshape(-1,3)

    ray = uv @ Ki.numpy().T

    ray = ray.reshape(1,-1,3).astype(np.float32)
    self.ray = torch.from_numpy(ray)
    self.u = torch.from_numpy(u.astype('float32'))
    self.v = torch.from_numpy(v.astype('float32'))

  def transform(self, xyz, R=None, t=None):
    if t is not None:
      bs = xyz.shape[0]
      xyz = xyz - t.reshape(bs,1,3)
    if R is not None:
      xyz = torch.bmm(xyz, R)
    return xyz

  def unproject(self, depth, R=None, t=None):
    self.ray = self.ray.to(depth.device)
    bs = depth.shape[0]

    xyz = depth.reshape(bs,-1,1) * self.ray
    xyz = self.transform(xyz, R, t)
    return xyz

  def project(self, xyz, R, t, return_ray_format= False):
    self.K = self.K.to(xyz.device)
    bs = xyz.shape[0]

    xyz = torch.bmm(xyz, R.transpose(1,2))
    xyz = xyz + t.reshape(bs,1,3)

    if return_ray_format:
      uv = xyz
    else:
      Kt = self.K.transpose(1,2).expand(bs,-1,-1)
      uv = torch.bmm(xyz, Kt)

    d = uv[:,:,2:3]

    # avoid division by zero
    uv = uv[:,:,:2] / (torch.nn.functional.relu(d) + 1e-12)
    return uv, d


  def tforward(self, depth0, R0, t0, R1, t1, return_ray_format= False):
    xyz = self.unproject(depth0, R0, t0)
    return self.project(xyz, R1, t1, return_ray_format)


class ProjectionDepthSimilarityLoss(ProjectionBaseLoss):
  '''
  Geometric Loss
  '''
  def __init__(self, *args, clamp=-1):
    super().__init__(*args)
    self.mod_name = 'ProjectionDepthSimilarityLoss'
    self.clamp = clamp

  def fwd(self, depth0, depth1, R0, t0, R1, t1):
    self.u = self.u.to(depth0.device)
    self.v = self.v.to(depth0.device)

    uv1, d1 = super().tforward(depth0, R0, t0, R1, t1)
    uv1 = uv1.view(-1, self.im_height, self.im_width, 2)
    d1 = d1.view(-1, 1, self.im_height, self.im_width)

    # Calculation rigid flow
    rigid_flow = uv1.clone()
    rigid_flow[..., 0] -= self.u
    rigid_flow[..., 1] -= self.v
    rigid_flow = rigid_flow.permute(0, 3, 1, 2)

    uv1[..., 0] = 2 * (uv1[..., 0] / (self.im_width-1) - 0.5)
    uv1[..., 1] = 2 * (uv1[..., 1] / (self.im_height-1) - 0.5)
    depth10 = torch.nn.functional.grid_sample(depth1, uv1, padding_mode='border', align_corners=True)

    diff = torch.abs(d1 - depth10)

    if self.clamp > 0:
      diff = torch.clamp(diff, 0, self.clamp)

    orig_mask = (diff.detach() < self.clamp).to('cpu').numpy().astype('float32')

    return diff.mean(), rigid_flow, orig_mask[0][0]

  def tforward(self, depth0, depth1, R0, t0, R1, t1):
    l0, rigid_flow0, orig_mask = self.fwd(depth0, depth1, R0, t0, R1, t1)
    l1, rigid_flow1, _ = self.fwd(depth1, depth0, R1, t1, R0, t0)

    with torch.no_grad():
      mask0 = self.generate_mask(rigid_flow0.detach(), rigid_flow1.detach())
      mask1 = self.generate_mask(rigid_flow1.detach(), rigid_flow0.detach())

    return l0+l1, rigid_flow0, rigid_flow1, mask0, mask1, orig_mask

  def generate_mask(self, flow0, flow1):
    uv1 = flow0.clone().permute(0, 2, 3, 1)
    uv1[..., 0] += self.u
    uv1[..., 1] += self.v
    uv1[..., 0] = 2 * (uv1[..., 0] / (self.im_width-1) - 0.5)
    uv1[..., 1] = 2 * (uv1[..., 1] / (self.im_height-1) - 0.5)
    flow0_proj = torch.nn.functional.grid_sample(flow1, uv1, padding_mode='border', align_corners=True)
    mask0 = ((flow0 + flow0_proj) ** 2).sum(dim=1) < 0.25 + 0.02 * ((flow0 ** 2).sum(dim=1) + (flow0_proj ** 2).sum(dim=1))
    mask0 = mask0.type(torch.float32).unsqueeze(1)
    return mask0


class Multi_Frame_Flow_Consistency_Loss(ProjectionBaseLoss):
  '''
  Flow Consistency Loss for Multi-Frame Inference
  '''

  def __init__(self, *args, clamp=-1):
    super().__init__(*args)
    self.mod_name = 'Multi_Frame_Flow_Consistency_Loss'
    self.clamp = clamp

  def fwd(self, depth0, depth1, R0, t0, R1, t1, flow0, flow1, amb0, amb1, primary_depth1):
    self.u = self.u.to(depth0.device)
    self.v = self.v.to(depth0.device)

    uv1, d1 = super().tforward(depth0, R0, t0, R1, t1)
    uv1 = uv1.view(-1, self.im_height, self.im_width, 2)
    d1 = d1.view(-1, 1, self.im_height, self.im_width)

    uv1_flow = flow0.permute(0, 2, 3, 1).clone()
    uv1_flow[..., 0] += self.u
    uv1_flow[..., 1] += self.v

    uv1_flow[..., 0] = 2 * (uv1_flow[..., 0] / (self.im_width - 1) - 0.5)
    uv1_flow[..., 1] = 2 * (uv1_flow[..., 1] / (self.im_height - 1) - 0.5)
    depth10 = torch.nn.functional.grid_sample(depth1, uv1_flow, padding_mode='zeros', align_corners=True)

    diff = torch.abs(d1 - depth10)

    with torch.no_grad():
      flow10 = torch.nn.functional.grid_sample(flow1.detach(), uv1_flow.detach(), padding_mode='zeros', align_corners=True)
      fb_mask = ((flow0.detach() + flow10) ** 2).sum(dim=1) < 0.5 + 0.02 * (
                (flow0.detach() ** 2).sum(dim=1) + (flow10 ** 2).sum(dim=1))
      fb_mask = fb_mask.type(torch.float32).unsqueeze(1)

      amb10 = torch.nn.functional.grid_sample(amb1.detach(), uv1_flow.detach(), padding_mode='zeros', align_corners=True)
      vc_mask = (((amb0 - amb10).abs()).mean(dim=1, keepdim=True) < 0.01).type(torch.float32)

      uv0, d0 = super().tforward(primary_depth1.detach(), R1.detach(), t1.detach(), R0.detach(), t0.detach(), return_ray_format= False)
      uv0 = uv0.view(-1, self.im_height, self.im_width, 2).permute(0, 3, 1, 2)
      warped_uv0 = torch.nn.functional.grid_sample(uv0.detach(), uv1_flow.detach(), padding_mode='zeros', align_corners=True)
      self_uv = torch.stack([self.u, self.v], dim= 0).unsqueeze(0)
      rf_mask = (((warped_uv0 - self_uv) ** 2).sum(dim = 1, keepdim=True) < 1).type(torch.float32)

      loss_mask = fb_mask * vc_mask * rf_mask

    diff = (diff * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    return diff

  def tforward(self, depth0, depth1, R0, t0, R1, t1, flow0, flow1, amb0, amb1, primary_depth0, primary_depth1):
    l0 = self.fwd(depth0, depth1, R0, t0, R1, t1, flow0, flow1, amb0, amb1, primary_depth1)
    l1 = self.fwd(depth1, depth0, R1, t1, R0, t0, flow1, flow0, amb1, amb0, primary_depth0)

    return l0 + l1

class Single_Frame_Flow_Consistency_Loss(ProjectionBaseLoss):
  '''
  Flow Consistency Loss for Single Frame Inference
  '''

  def __init__(self, *args, clamp=-1):
    super().__init__(*args)
    self.mod_name = 'Single_Frame_Flow_Consistency_Loss'
    self.clamp = clamp

  def fwd(self, depth0, depth1, R0, t0, R1, t1, flow0, flow1, amb0, amb1):
    self.u = self.u.to(depth0.device)
    self.v = self.v.to(depth0.device)

    uv1, d1 = super().tforward(depth0, R0, t0, R1, t1)
    uv1 = uv1.view(-1, self.im_height, self.im_width, 2)
    d1 = d1.view(-1, 1, self.im_height, self.im_width)

    uv1_flow = flow0.permute(0, 2, 3, 1).clone()
    uv1_flow[..., 0] += self.u
    uv1_flow[..., 1] += self.v

    uv1_flow[..., 0] = 2 * (uv1_flow[..., 0] / (self.im_width - 1) - 0.5)
    uv1_flow[..., 1] = 2 * (uv1_flow[..., 1] / (self.im_height - 1) - 0.5)
    depth10 = torch.nn.functional.grid_sample(depth1, uv1_flow, padding_mode='zeros', align_corners=True)

    diff = torch.abs(d1 - depth10)

    if self.clamp > 0:
      diff = torch.clamp(diff, 0, self.clamp)

    orig_mask = (diff.detach() < self.clamp).to('cpu').numpy().astype('float32')

    with torch.no_grad():
      flow10 = torch.nn.functional.grid_sample(flow1.detach(), uv1_flow.detach(), padding_mode='zeros', align_corners=True)
      fb_mask = ((flow0.detach() + flow10) ** 2).sum(dim=1) < 0.5 + 0.02 * (
                (flow0.detach() ** 2).sum(dim=1) + (flow10 ** 2).sum(dim=1))
      fb_mask = fb_mask.type(torch.float32).unsqueeze(1)

      amb10 = torch.nn.functional.grid_sample(amb1.detach(), uv1_flow.detach(), padding_mode='zeros', align_corners=True)
      vc_mask = (((amb0 - amb10).abs()).mean(dim=1, keepdim=True) < 0.01).type(torch.float32)

      loss_mask = fb_mask * vc_mask

    diff = (diff * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    return diff, loss_mask, orig_mask[0][0]

  def tforward(self, depth0, depth1, R0, t0, R1, t1, flow0, flow1, amb0, amb1):
    l0, mask0, orig_mask = self.fwd(depth0, depth1, R0, t0, R1, t1, flow0, flow1, amb0, amb1)
    l1, mask1, _ = self.fwd(depth1, depth0, R1, t1, R0, t0, flow1, flow0, amb1, amb0)

    return l0 + l1, mask0, mask1, orig_mask

class LCN(TimedModule):
  '''
  Local Contract Normalization
  '''
  def __init__(self, radius, epsilon):
    super().__init__(mod_name='LCN')
    self.box_conv = torch.nn.Sequential(
        torch.nn.ReflectionPad2d(radius),
        torch.nn.Conv2d(1, 1, kernel_size=2*radius+1, bias=False)
    )
    self.box_conv[1].weight.requires_grad=False
    self.box_conv[1].weight.fill_(1.)

    self.epsilon = epsilon
    self.radius = radius

  def tforward(self, data):
    boxs = self.box_conv(data)

    avgs = boxs / (2*self.radius+1)**2
    boxs_n2 = boxs**2
    boxs_2n = self.box_conv(data**2)

    stds = torch.sqrt(torch.clamp(boxs_2n / (2*self.radius+1)**2 - avgs**2 + 1e-6, min=0))
    stds = stds + self.epsilon

    return (data - avgs) / stds, stds



class SobelFilter(TimedModule):
  '''
  Sobel Filter
  '''
  def __init__(self, norm=False, ksize = 5):
    super(SobelFilter, self).__init__(mod_name='SobelFilter')
    self.ksize = ksize
    if self.ksize == 5:
      kx = np.array([[-5, -4, 0, 4, 5],
                     [-8, -10, 0, 10, 8],
                     [-10, -20, 0, 20, 10],
                     [-8, -10, 0, 10, 8],
                     [-5, -4, 0, 4, 5]])/240.0
    elif self.ksize == 3:
      kx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])/8.0

    ky = kx.copy().transpose(1,0)

    self.conv_x=torch.nn.Conv2d(1, 1, kernel_size=self.ksize, stride=1, padding=0, bias=False)
    self.conv_x.weight=torch.nn.Parameter(torch.from_numpy(kx).float().unsqueeze(0).unsqueeze(0))

    self.conv_y=torch.nn.Conv2d(1, 1, kernel_size=self.ksize, stride=1, padding=0, bias=False)
    self.conv_y.weight=torch.nn.Parameter(torch.from_numpy(ky).float().unsqueeze(0).unsqueeze(0))

    self.norm=norm

  def tforward(self,x):
    if self.ksize == 5:
      x = F.pad(x, (2,2,2,2), "replicate")
    elif self.ksize == 3:
      x = F.pad(x, (1,1,1,1), "replicate")
    gx = self.conv_x(x)
    gy = self.conv_y(x)
    if self.norm:
      return torch.sqrt(gx**2 + gy**2 + 1e-8)
    else:
      return torch.cat((gx, gy), dim=1)