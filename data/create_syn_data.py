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

import argparse

import numpy as np
import pickle
from pathlib import Path
import time
import json
import cv2
import collections
import sys
import h5py
import os

sys.path.append('../')
import co
from data_manipulation import get_rotation_matrix, read_pattern_file, post_process

config_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'config.json'))
with open(config_path) as fp:
    config = json.load(fp)
    CTD_path = Path(config['CTD_DIR'])
    sys.path.append(str(CTD_path / 'data'))
    sys.path.append(str(CTD_path / 'renderer'))

from lcn import lcn
from cyrender import PyRenderInput, PyCamera, PyShader, PyRenderer

def get_objs(shapenet_dir, obj_classes, num_perclass=100):
    shapenet = {'chair': '03001627',
                'airplane': '02691156',
                'car': '02958343',
                'watercraft': '04530566'}

    obj_paths = []
    for cls in obj_classes:
        if cls not in shapenet.keys():
            raise Exception('unknown class name')
        ids = shapenet[cls]
        obj_path = sorted(Path(f'{shapenet_dir}/{ids}').glob('**/models/*.obj'))
        obj_paths += obj_path[:num_perclass]
    print(f'found {len(obj_paths)} object paths')

    objs = []
    for obj_path in obj_paths:
        print(f'load {obj_path}')
        v, f, _, n = co.io3d.read_obj(obj_path)
        diffs = v.max(axis=0) - v.min(axis=0)
        v /= (0.5 * diffs.max())
        v -= (v.min(axis=0) + 1)
        f = f.astype(np.int32)
        objs.append((v, f, n))
    print(f'loaded {len(objs)} objects')

    return objs


def get_mesh(rng, min_z=0):
    # set up background board
    verts, faces, normals, colors = [], [], [], []
    v, f, n = co.geometry.xyplane(z=0, interleaved=True)
    v[:, 2] += -v[:, 2].min() + rng.uniform(3, 5)
    v[:, :2] *= 5e2
    v[:, 2] = np.mean(v[:, 2]) + (v[:, 2] - np.mean(v[:, 2])) * 5e2
    c = np.empty_like(v)
    c[:] = rng.uniform(0, 1, size=(3,)).astype(np.float32)
    verts.append(v)
    faces.append(f)
    normals.append(n)
    colors.append(c)

    # randomly sample 4 foreground objects for each scene
    for shape_idx in range(4):
        v, f, n = objs[rng.randint(0, len(objs))]
        v, f, n = v.copy(), f.copy(), n.copy()

        s = rng.uniform(0.25, 1)
        v *= s
        R = co.geometry.rotm_from_quat(co.geometry.quat_random(rng=rng))
        v = v @ R.T
        n = n @ R.T
        v[:, 2] += -v[:, 2].min() + min_z + rng.uniform(0.5, 3)
        v[:, :2] += rng.uniform(-1, 1, size=(1, 2))

        c = np.empty_like(v)
        c[:] = rng.uniform(0, 1, size=(3,)).astype(np.float32)

        verts.append(v.astype(np.float32))
        faces.append(f)
        normals.append(n)
        colors.append(c)

    verts, faces = co.geometry.stack_mesh(verts, faces)
    normals = np.vstack(normals).astype(np.float32)
    colors = np.vstack(colors).astype(np.float32)
    return verts, faces, colors, normals


def create_data(pattern_type, out_root, idx, n_samples, imsize_proj, imsize, pattern, K_proj, K, K_processed, baseline, blend_im, noise,
                track_length=4):
    tic = time.time()
    rng = np.random.RandomState()

    rng.seed(idx)

    verts, faces, colors, normals = get_mesh(rng)
    data = PyRenderInput(verts=verts.copy(), colors=colors.copy(), normals=normals.copy(), faces=faces.copy())
    print(f'loading mesh for sample {idx + 1}/{n_samples} took {time.time() - tic}[s]')

    # let the camera point to the center
    center = np.array([0, 0, 3], dtype=np.float32)

    basevec = np.array([-baseline, 0, 0], dtype=np.float32)
    unit = np.array([0, 0, 1], dtype=np.float32)

    cam_x_ = rng.uniform(-0.2, 0.2)
    cam_y_ = rng.uniform(-0.2, 0.2)
    cam_z_ = rng.uniform(-0.2, 0.2)

    ret = collections.defaultdict(list)
    blend_im_rnd = np.clip(blend_im + rng.uniform(-0.1, 0.1), 0, 1)

    # capture the same static scene from different view points as a track
    for ind in range(track_length):

        cam_x = cam_x_ + rng.uniform(-0.1, 0.1)
        cam_y = cam_y_ + rng.uniform(-0.1, 0.1)
        cam_z = cam_z_ + rng.uniform(-0.1, 0.1)

        tcam = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        if np.linalg.norm(tcam[0:2]) < 1e-9:
            Rcam = np.eye(3, dtype=np.float32)
        else:
            Rcam = get_rotation_matrix(center, center - tcam)

        tproj = tcam + basevec
        Rproj = Rcam

        ret['R'].append(Rcam)
        ret['t'].append(tcam)

        fx_proj = K_proj[0, 0]
        fy_proj = K_proj[1, 1]
        px_proj = K_proj[0, 2]
        py_proj = K_proj[1, 2]
        im_height_proj = imsize_proj[0]
        im_width_proj = imsize_proj[1]
        proj = PyCamera(fx_proj, fy_proj, px_proj, py_proj, Rproj, tproj, im_width_proj, im_height_proj)

        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        im_height = imsize[0]
        im_width = imsize[1]
        cam = PyCamera(fx, fy, px, py, Rcam, tcam, im_width, im_height)

        shader = PyShader(0.5, 1.5, 0.0, 10)
        pyrenderer = PyRenderer(cam, shader, engine='gpu')
        if args.pattern_type == 'default':
            pyrenderer.mesh_proj(data, proj, pattern.copy(), d_alpha=0, d_beta=0.0)
        else:
            pyrenderer.mesh_proj(data, proj, pattern.copy(), d_alpha=0, d_beta=0.35)

        # get the reflected laser pattern $R$
        im = pyrenderer.color().copy()
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        focal_length = K_processed[0, 0]
        depth = pyrenderer.depth().copy()
        disp = baseline * focal_length / depth

        # get the ambient image $A$
        ambient = pyrenderer.normal().copy()
        ambient = cv2.cvtColor(ambient, cv2.COLOR_RGB2GRAY)

        # get the noise free IR image $J$
        im = blend_im_rnd * im + (1 - blend_im_rnd) * ambient
        ret['ambient'].append(post_process(pattern_type, ambient)[None].astype(np.float32))

        # get the gradient magnitude of the ambient image $|\nabla A|$
        ambient = ambient.astype(np.float32)
        sobelx = cv2.Sobel(ambient, cv2.CV_32F, 1, 0, ksize=5)
        sobely = cv2.Sobel(ambient, cv2.CV_32F, 0, 1, ksize=5)
        grad = np.sqrt(sobelx ** 2 + sobely ** 2)
        grad = np.maximum(grad - 0.8, 0.0)  # parameter

        # get the local contract normalized grad LCN($|\nabla A|$)
        grad_lcn, grad_std = lcn.normalize(grad, 5, 0.1)
        grad_lcn = np.clip(grad_lcn, 0.0, 1.0)  # parameter
        ret['grad'].append(post_process(pattern_type, grad_lcn)[None].astype(np.float32))

        ret['im'].append(post_process(pattern_type, im)[None].astype(np.float32))
        ret['disp'].append(post_process(pattern_type, disp)[None].astype(np.float32))

    for key in ret.keys():
        ret[key] = np.stack(ret[key], axis=0)

    # save to files
    out_dir = out_root / f'{idx:08d}'
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f'frames.hdf5'
    with h5py.File(out_path, "w") as f:
        for k, val in ret.items():
            # f.create_dataset(k, data=val, compression="lzf")
            f.create_dataset(k, data=val)

    print(f'create sample {idx + 1}/{n_samples} took {time.time() - tic}[s]')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('pattern_type',
                        help='Select the pattern file for projecting dots',
                        default='default',
                        choices=['default', 'kinect', 'real'], type=str)
    args = parser.parse_args()

    np.random.seed(42)

    # output directory
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    with open(config_path) as fp:
        config = json.load(fp)
        data_root = Path(config['DATA_DIR'])
        shapenet_root = config['SHAPENET_DIR']

    out_root = data_root
    out_root.mkdir(parents=True, exist_ok=True)

    # load shapenet models
    obj_classes = ['chair']
    objs = get_objs(shapenet_root, obj_classes)

    # camera parameters
    if args.pattern_type == 'real':
        fl_proj = 1112.1806640625
        fl = 1112.1806640625
        imsize_proj = (1280, 1080)
        imsize = (1280, 1080)
        imsize_processed = (512, 432)
        K_proj = np.array([[fl_proj, 0, 517.0896606445312], [0, fl_proj, 649.6329956054688], [0, 0, 1]], dtype=np.float32)
        K = np.array([[fl, 0, 517.0896606445312], [0, fl, 649.6329956054688], [0, 0, 1]], dtype=np.float32)
        baseline = 0.0246
        blend_im = 0.6
        noise = 0
    else:
        fl_proj = 1582.06005876
        fl = 435.2
        imsize_proj = (4096, 4096)
        imsize = (512, 432)
        imsize_processed = (512, 432)
        K_proj = np.array([[fl_proj, 0, 2047.5], [0, fl_proj, 2047.5], [0, 0, 1]], dtype=np.float32)
        K = np.array([[fl, 0, 216], [0, fl, 256], [0, 0, 1]], dtype=np.float32)
        baseline = 0.025
        blend_im = 0.6
        noise = 0

    # capture the same static scene from different view points as a track
    track_length = 4

    # load pattern image
    pattern = read_pattern_file(args.pattern_type, imsize_proj)

    x_cam = np.arange(0, imsize[1])
    y_cam = np.arange(0, imsize[0])
    x_mesh, y_mesh = np.meshgrid(x_cam, y_cam)
    x_mesh_f = np.reshape(x_mesh, [-1])
    y_mesh_f = np.reshape(y_mesh, [-1])

    grid_points = np.stack([x_mesh_f, y_mesh_f, np.ones_like(x_mesh_f)], axis=0)
    grid_points_mapped = K_proj.dot(np.linalg.inv(K).dot(grid_points))
    grid_points_mapped = grid_points_mapped / grid_points_mapped[2, :]

    x_map = np.reshape(grid_points_mapped[0, :], x_mesh.shape)
    y_map = np.reshape(grid_points_mapped[1, :], y_mesh.shape)
    x_map, y_map = x_map.astype('float32'), y_map.astype('float32')
    mapped_pattern = cv2.remap(pattern, x_map, y_map, cv2.INTER_LINEAR)

    pattern_processed, K_processed = post_process(args.pattern_type, mapped_pattern, K)
    # write settings to file
    settings = {
        'imsize': imsize_processed,
        'pattern': pattern_processed,
        'baseline': baseline,
        'K': K_processed,
    }
    out_path = out_root / f'settings.pkl'
    print(f'write settings to {out_path}')
    with open(str(out_path), 'wb') as f:
        pickle.dump(settings, f, pickle.HIGHEST_PROTOCOL)

    # start the job
    n_samples = 2 ** 10 + 2 ** 13
    # n_samples = 2048
    for idx in range(n_samples):
        parameters = (
        args.pattern_type, out_root, idx, n_samples, imsize_proj, imsize, pattern, K_proj, K, K_processed, baseline, blend_im, noise, track_length)
        create_data(*parameters)