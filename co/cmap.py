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

_color_map_errors = np.array([
    [149, 54, 49],  # 0: log2(x) = -infinity
    [180, 117, 69],  # 0.0625: log2(x) = -4
    [209, 173, 116],  # 0.125: log2(x) = -3
    [233, 217, 171],  # 0.25: log2(x) = -2
    [248, 243, 224],  # 0.5: log2(x) = -1
    [144, 224, 254],  # 1.0: log2(x) = 0
    [97, 174, 253],  # 2.0: log2(x) = 1
    [67, 109, 244],  # 4.0: log2(x) = 2
    [39, 48, 215],  # 8.0: log2(x) = 3
    [38, 0, 165],  # 16.0: log2(x) = 4
    [38, 0, 165]  # inf: log2(x) = inf
]).astype(float)


def color_error_image(errors, scale=1.2, log_scale=0.25, mask=None, BGR=True):
    """
    Color an input error map.

    Arguments:
        errors -- HxW numpy array of errors
        [scale=1] -- scaling the error map (color change at unit error)
        [mask=None] -- zero-pixels are masked white in the result
        [BGR=True] -- toggle between BGR and RGB

    Returns:
        colored_errors -- HxWx3 numpy array visualizing the errors
    """

    errors_flat = errors.flatten()
    errors_color_indices = np.clip(np.log2(errors_flat / scale + 1e-5) / log_scale + 5, 0, 9)
    i0 = np.floor(errors_color_indices).astype(int)
    f1 = errors_color_indices - i0.astype(float)
    colored_errors_flat = _color_map_errors[i0, :] * (1 - f1).reshape(-1, 1) + _color_map_errors[i0 + 1,
                                                                               :] * f1.reshape(-1, 1)

    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 255

    if not BGR:
        colored_errors_flat = colored_errors_flat[:, [2, 1, 0]]

    return colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.int)


_color_map_depths = np.array([
    [0, 0, 0],  # 0.000
    [0, 0, 255],  # 0.114
    [255, 0, 0],  # 0.299
    [255, 0, 255],  # 0.413
    [0, 255, 0],  # 0.587
    [0, 255, 255],  # 0.701
    [255, 255, 0],  # 0.886
    [255, 255, 255],  # 1.000
    [255, 255, 255],  # 1.000
]).astype(float)
_color_map_bincenters = np.array([
    0.0,
    0.114,
    0.299,
    0.413,
    0.587,
    0.701,
    0.886,
    1.000,
    2.000,  # doesn't make a difference, just strictly higher than 1
])


def color_depth_map(depths, scale=None):
    """
    Color an input depth map.

    Arguments:
        depths -- HxW numpy array of depths
        [scale=None] -- scaling the values (defaults to the maximum depth)

    Returns:
        colored_depths -- HxWx3 numpy array visualizing the depths
    """

    if scale is None:
        scale = depths.max()

    values = np.clip(depths.flatten() / scale, 0, 1)
    # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[
        lower_bin + 1] * alphas.reshape(-1, 1)
    return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)

# from utils.debug import save_color_numpy
# save_color_numpy(color_depth_map(np.matmul(np.ones((100,1)), np.arange(0,1200).reshape(1,1200)), scale=1000))
