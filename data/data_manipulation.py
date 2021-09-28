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

import numpy as np
import cv2

def read_pattern_file(pattern_type, pattern_size):
    if pattern_type == 'default':
        pattern_path = 'default_pattern.png'
    elif pattern_type == 'kinect':
        pattern_path = 'kinect_pattern.png'
    elif pattern_type == 'real':
        pattern_path = 'real_pattern.png'

    pattern = cv2.imread(pattern_path)
    pattern = pattern.astype(np.float32)
    pattern /= 255

    if pattern.ndim == 2:
        pattern = np.stack([pattern for idx in range(3)], axis=2)

    if pattern_type == 'default':
        pattern = np.rot90(np.flip(pattern, axis=1))
    elif pattern_type == 'kinect':
        min_dim = min(pattern.shape[0:2])
        start_h = (pattern.shape[0] - min_dim) // 2
        start_w = (pattern.shape[1] - min_dim) // 2
        pattern = pattern[start_h:start_h + min_dim, start_w:start_w + min_dim]
        pattern = cv2.resize(pattern, pattern_size, interpolation=cv2.INTER_LINEAR)

    return pattern

def get_rotation_matrix(v0, v1):
  v0 = v0/np.linalg.norm(v0)
  v1 = v1/np.linalg.norm(v1)
  v = np.cross(v0,v1)
  c = np.dot(v0,v1)
  s = np.linalg.norm(v)
  I = np.eye(3)
  vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
  k = np.matrix(vXStr)
  r = I + k + k @ k * ((1 -c)/(s**2))
  return np.asarray(r.astype(np.float32))

def post_process(pattern_type, im, K = None):
    if pattern_type == 'real':
        im_processed = im[128:-128, 108:-108, ...].copy()
        im_processed = cv2.resize(im_processed, (432, 512), interpolation=cv2.INTER_LINEAR)

        if K is not None:
            K_processed = K.copy()

            K_processed[0, 0] = K_processed[0, 0] / 2
            K_processed[1, 1] = K_processed[1, 1] / 2

            K_processed[0, 2] = (K_processed[0, 2] - 108) / 2
            K_processed[1, 2] = (K_processed[1, 2] - 128) / 2

            return im_processed, K_processed
        else:
            return im_processed
    else:
        if K is not None:
            return im, K
        else:
            return im

def augment_image(img,rng, amb=None, disp=None, primary_disp= None, sgm_disp= None, grad=None,max_shift=64,max_blur=1.5,max_noise=10.0,max_sp_noise=0.001):

    # get min/max values of image
    min_val = np.min(img)
    max_val = np.max(img)
    
    # init augmented image
    img_aug = img
    amb_aug = amb

    # init disparity correction map
    disp_aug = disp
    primary_disp_aug = primary_disp
    sgm_disp_aug = sgm_disp
    grad_aug = grad

    # apply affine transformation
    if max_shift>1:
        
        # affine parameters
        rows,cols = img.shape
        shear = 0
        shift = 0
        shear_correction = 0
        if rng.uniform(0,1)<0.75: shear = rng.uniform(-max_shift,max_shift) # shear with 75% probability
        else:                     shift = rng.uniform(-max_shift/2,max_shift)          # shift with 25% probability
        if shear<0:               shear_correction = -shear
        
        # affine transformation
        a = shear/float(rows)
        b = shift+shear_correction
        
        # warp image
        T = np.float32([[1,a,b],[0,1,0]])                
        img_aug = cv2.warpAffine(img_aug,T,(cols,rows))
        if amb is not None:
            amb_aug = cv2.warpAffine(amb_aug,T,(cols,rows))
        if grad is not None:
          grad_aug = cv2.warpAffine(grad,T,(cols,rows))
        
        # disparity correction map
        col = a*np.array(range(rows))+b
        disp_delta = np.tile(col,(cols,1)).transpose()
        if disp is not None:
          disp_aug = cv2.warpAffine(disp+disp_delta,T,(cols,rows))
        if primary_disp is not None:
          primary_disp_aug = cv2.warpAffine(primary_disp+disp_delta,T,(cols,rows))
        if sgm_disp is not None:
          sgm_disp_aug = cv2.warpAffine(sgm_disp+disp_delta,T,(cols,rows))

    # gaussian smoothing
    if rng.uniform(0,1)<0.5:
        img_aug = cv2.GaussianBlur(img_aug,(5,5),rng.uniform(0.2,max_blur))
        if amb is not None:
            amb_aug = cv2.GaussianBlur(amb_aug,(5,5),rng.uniform(0.2,max_blur))

    # per-pixel gaussian noise
    img_aug = img_aug + rng.randn(*img_aug.shape)*rng.uniform(0.0,max_noise)/255.0
    if amb is not None:
        amb_aug = amb_aug + rng.randn(*amb_aug.shape)*rng.uniform(0.0,max_noise)/255.0

    # salt-and-pepper noise
    if rng.uniform(0,1)<0.5:
        ratio=rng.uniform(0.0,max_sp_noise)
        img_shape = img_aug.shape
        img_aug = img_aug.flatten()
        coord = rng.choice(np.size(img_aug), int(np.size(img_aug)*ratio))
        img_aug[coord] = max_val
        coord = rng.choice(np.size(img_aug), int(np.size(img_aug)*ratio))
        img_aug[coord] = min_val
        img_aug = np.reshape(img_aug, img_shape)
        
    # clip intensities back to [0,1]
    img_aug = np.maximum(img_aug,0.0)
    img_aug = np.minimum(img_aug,1.0)

    if amb is not None:
        amb_aug = np.maximum(amb_aug,0.0)
        amb_aug = np.minimum(amb_aug,1.0)

    # return image
    return img_aug, amb_aug, disp_aug, primary_disp_aug, sgm_disp_aug, grad_aug
