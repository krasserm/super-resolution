# Single Image Super-Resolution for Keras

A Keras-based implementation of

- [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR). Winner 
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge.
- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR). Winner 
  of the [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) super-resolution challenge.

WDSR models are trained with *weight normalization* as described in

- [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)

A modified Adam optimizer that performs weight normalization is available 
[here](https://github.com/krasserm/weightnorm/tree/master/keras_2) and has been copied to 
[weightnorm.py](optimizer/weightnorm.py) in this repository. 

## Setup

This project requires Python 3.6 or higher.

1. Create a new [virtual environment](https://docs.python.org/3/tutorial/venv.html) or 
   [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) and activate that environment.

2. Run `pip install -r requirements-gpu.txt` if you have a GPU or `pip install -r requirements-gpu.txt` otherwise.


## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/13YjKmP5O8NK_E_dFlK-34Okn1IIM9c58).
Each directory contains a model together with the training settings. At the moment, only baseline models available
and I'll upload bigger models later. You can also train models yourself as described further below. The following 
table gives an overview of available pre-trained models and their performance (PSNR) on the DIV2K benchmark:

 
<table>
    <tr>
        <th>Model</th>
        <th>Scale</th>
        <th>Parameters</th>
        <th>PSNR (DIV2K)</th>
        <th>Training</th>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1Vr_eLXnNA7H6zNWmEFKOBv4-xvOBt5iu">wdsr-b-8-x2</a></td>
        <td>x2</td>
        <td>0.89M</td>
        <td>34.539</td>
        <td><a href="https://drive.google.com/open?id=1VL4i4i1XuMy65wbq8fiWOOfMNziRqmdE">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1CSdinKy9E3B4dm-lp7O_W-MYXp0GoB9g">wdsr-b-8-x3</a></td>
        <td>x3</td>
        <td>0.89M</td>
        <td>30.865</td>
        <td><a href="https://drive.google.com/open?id=1B2w-ZSlD96RkCQ5C_JbQEDrdIMez7y3D">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1WCpIY9G-9fL9cTa3We9ry3hm-ePT58b_">wdsr-b-8-x4</a></td>
        <td>x4</td>
        <td>0.90M</td>
        <td>28.912</td>
        <td><a href="https://drive.google.com/open?id=1jgQfwGR_HVqVUjQqkvHCDhHowvTBmP5_">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1tp7r_oUf8Ohd9q-ouGApS7qNtqg1IRLt">wdsr-a-8-x2</a></td>
        <td>x2</td>
        <td>0.60M</td>
        <td>34.469</td>
        <td><a href="https://drive.google.com/open?id=1hnL23k9_UYvGeAhY2nWOMM1rP2k-t8d-">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1ujCCDTJIheyGW-2wLU96tH13dGMEg84i">edsr-8-x2</a><sup>*)</sup></td>
        <td>x2</td>
        <td>0.78M</td>
        <td>34.414</td>
        <td><a href="https://drive.google.com/open?id=1x8EjZxvTt0WO4zSdLDgBkKep3jYntrWc">settings</a></td>
    </tr>
</table>

<sup>*)</sup> This is a smaller EDSR baseline model than that described in the EDSR paper (has only 8 residual blocks 
instead of 16).

## Demo

The following example super-resolves images in directory [`./demo`](demo) with factor x4 using the downloaded 
[wdsr-b-8-x4](https://drive.google.com/open?id=1WCpIY9G-9fL9cTa3We9ry3hm-ePT58b_) **baseline model** and writes the 
results to directory [`./output`](output):

    python demo.py -i ./demo -o ./output --model=./wdsr-b-8-x4-psnr-28.9121.h5
    
The following figures also compare the super-resolution results (SR) with the corresponding low-resolution (LR) and 
high-resolution (HR) images and an x4 resize with bicubic interpolation. The demo images were cropped from images in 
the DIV2K validation set. 

![0829](docs/demo-0829.png)

![0851](docs/demo-0851.png)

## DIV2K Dataset

If you want to train and/or evaluate models, you need to download the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
and extract the downloaded archives to a directory of your choice (`my_dir` in the following example). You'll later 
refer to this directory with the `--dataset` command line option. The resulting directory structure should look like
  
    my_dir
      DIV2K_train_HR
        DIV2K_train_LR_bicubic
          X2
          X3
          X4
        DIV2K_train_LR_unknown
          X2
          X3
          X4
        DIV2K_valid_HR
        DIV2K_valid_LR_bicubic
          ...
        DIV2K_valid_LR_unknown
          ...
          
You only need to download DIV2K archives for those downgrade operators (unknown, bicubic) and super-resolution scales
(x2, x3, x4) that you'll actually use for training.

## Training

Coming soon ...

## Evaluation

Coming soon ...
