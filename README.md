
> # [ICCV 2021] DepthInSpace: Exploitation and Fusion of Multiple Frames of a Video for Structured-Light Depth Estimation <br>
> Mohammad Mahdi Johari, Camilla Carta, François Fleuret <br>
> [Project Page](https://www.idiap.ch/paper/depthinspace/) | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Johari_DepthInSpace_Exploitation_and_Fusion_of_Multiple_Video_Frames_for_Structured-Light_ICCV_2021_paper.html)

## Dependencies

The network training/evaluation code is based on `PyTorch` and is  tested in the following environment:
```
Python==3.8.6
PyTorch==1.7.0
CUDA==10.1
```

All required packages can be installed with `anaconda`:
```
conda install --file requirements.txt -c pytorch -c conda-forge
```

### External Libraries
To train and evaluate our method on synthetic datasets, we use the structured light renderer provided by [Connecting the Dots](https://github.com/autonomousvision/connecting_the_dots).
It can be used to render a virtual scene (arbitrary triangle mesh) with the structured light pattern projected from a customizable projector location.
Furthermore, Our models use some custom layers provided in [Connecting the Dots](https://github.com/autonomousvision/connecting_the_dots).
First, download [ShapeNet V2](https://www.shapenet.org/) and correct `SHAPENET_DIR` in `config.json` accordingly.
Then, to install these dependencies, use the following instructions and set `CTD_DIR` in `config.json` to the path of the cloned [Connecting the Dots](https://github.com/autonomousvision/connecting_the_dots) repository:

```
git clone https://github.com/autonomousvision/connecting_the_dots.git
cd connecting_the_dots
cd renderer
make
cd ..
cd data/lcn
python setup.py build_ext --inplace
cd ../..
cd torchext
python setup.py build_ext --inplace
cd ..
```

As a preprocessing step, you need to execute [LiteFlowNet](https://github.com/sniklaus/pytorch-liteflownet) software on the data before running our models. To this end, clone [our forked copy of pytorch-liteflownet](https://github.com/MohammadJohari/pytorch-liteflownet) with the following command and set `LITEFLOWNET_DIR` in `config.json` accordingly.
```
git clone https://github.com/MohammadJohari/pytorch-liteflownet.git
```
Make sure you comply with the [license](https://github.com/twhui/LiteFlowNet#license-and-citation) terms of LiteFlowNet's original paper.
However, DepthInSpace models are compatible with any external optical flow library.
To use DepthInSpace with other optical flow libraries, you need to modify the code in `data/presave_optical_flow_data.py` and make it compatible with your optical flow model of choice.

## Running


### Creating Synthetic Data
The synthetic data will be generated and saved to `DATA_DIR` in `config.json`.
In order to generate the data with the `default` projection dot pattern, change directory to `data` and run

```
python create_syn_data.py default
```

Other available options for the dot pattern are: `kinect` and `real`, where `real` is the real observed dot pattern in our experiments.

### Pre-Saving Optical Flow Data
Before training, optical flow predictions from LiteFlowNet should be pre-saved. To do so, make sure the `DATA_DIR` in `config.json` is correct and run the following command in `data` directory

```
python presave_optical_flow_data.py
```

### Training DIS-SF
Note that the weights and state of training of our networks are saved in `OUTPUT_DIR` in `config.json`.
For training the DIS-SF network with an arbitrary batch size (e.g. 8), you can run

```
python train_val.py --train_batch_size 8 --architecture single_frame
```

### Training DIS-MF
before training the DIS-MF network, the DIS-SF network must have been trained and its outputs must have been pre-saved.
Make sure the `DATA_DIR` in `config.json` is correct and the trained weights of the DIS-SF network are available in `OUTPUT_DIR` in `config.json`. 
Then, you can pre-save the outputs of an specific epoch (e.g. 100) of the DIS-SF network by running the following command in `data` directory

```
python presave_disp.py single_frame --epoch 100
```

You can then train the DIS-MF network with an arbitrary batch size (e.g. 4) by running

```
python train_val.py --train_batch_size 4 --architecture multi_frame
```
The DIS-MF network can be trained with batch size of 4 on a device with 24 Gigabytes of GPU memory.

### Training DIS-FTSF
before training the DIS-FTSF network, the DIS-MF network must have been trained and its outputs must have been pre-saved.
Make sure the `DATA_DIR` in `config.json` is correct and the trained weights of the DIS-MF network are available in `OUTPUT_DIR` in `config.json`. 
Then, you can pre-save the outputs of an specific epoch (e.g. 50) of the DIS-MF network by running the following command in `data` directory

```
python presave_disp.py multi_frame --epoch 50
```

You can then train the DIS-FTSF network with an arbitrary batch size (e.g. 8) by running

```
python train_val.py --train_batch_size 8 --architecture single_frame --use_pseudo_gt True
```

### Evaluating the Networks
To evaluate a specific checkpoint of a specific network, e.g. the 50th epoch of the DIS-MF network, one can run 
```
python train_val.py --architecture multi_frame --cmd retest --epoch 50
```
### Training/Evaluating on Real Dataset
Our captured real dataset can be downloaded from [here](https://www.idiap.ch/en/dataset/depthinspace/index_html). Make sure to use `--data_type real` when you want to train or evaluate the models on our real dataset to use the same train/test split as in our paper.

### Pretrained Networks
The pretrained networks for synthetic datasets with different projection patterns and the real dataset can be found [here](https://drive.google.com/drive/folders/1uiSElbiQhXxag2VIpXy4lOoueoAEh1ak?usp=sharing). In order to use these networks, make sure `OUTPUT_DIR` in `config.json` corresponds to the proper pretrained directory. Then, use the following for the single-frame model:
```
python train_val.py --architecture single_frame --cmd retest --epoch 0
```
and the following for the multi-frame model:
```
python train_val.py --architecture multi_frame --cmd retest --epoch 0
```
Make sure to add `--data_type real` when you want to use the models on our real dataset to use the same train/test split as in our paper.

### Contact
You can contact the author through email: mohammad.johari At idiap.ch.

### License
All codes found in this repository are licensed under the [MIT License](LICENSE).

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{johari-et-al-2021,
  author = {Johari, M. and Carta, C. and Fleuret, F.},
  title = {DepthInSpace: Exploitation and Fusion of Multiple Video Frames for Structured-Light Depth Estimation},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2021}
}
```

### Acknowledgement
This work was supported by ams OSRAM.