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

from pathlib import Path
import os
import json
import pickle

if __name__ == '__main__':

    # output directory
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    with open(config_path) as fp:
        config = json.load(fp)
        data_root = Path(config['DATA_DIR'])
        liteflownet_path = Path(config['LITEFLOWNET_DIR'])

    script_path = str(liteflownet_path / 'run.py')
    data_path = str(data_root)
    os.environ['PYTHONPATH'] = str(liteflownet_path) + os.pathsep + str(liteflownet_path / 'correlation')
    os.system('python ' + script_path + ' --model default' + ' --data_path ' + data_path)
