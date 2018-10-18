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
results to directory `./output`:

    python demo.py -i ./demo -o ./output --model=./wdsr-b-8-x4-psnr-28.9121.h5
    
The following figures also compare the super-resolution results (SR) with the corresponding low-resolution (LR) and 
high-resolution (HR) images and an x4 resize with bicubic interpolation. The demo images were cropped from images in 
the DIV2K validation set. 

![0829](docs/demo-0829.png)

![0851](docs/demo-0851.png)

## DIV2K Dataset

If you want to train and/or evaluate models, you need to download the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
and extract the downloaded archives to a directory of your choice (`DIV2K` in the following example). You'll later 
refer to this directory with the `--dataset` command line option. The resulting directory structure should look like
  
    DIV2K
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

WDSR and EDSR models can be trained by running `train.py` with the command line options and profiles described in 
[`args.py`](args.py). For example, a WDSR-B baseline model with 8 residual blocks can be trained for scale x2 with

    python train.py --dataset ./DIV2K --outdir ./output --profile wdsr-b-8 --scale 2
    
The `--dataset` option sets the location of the DIV2K dataset and the `--output` option the output directory. 
Each training run creates a timestamped sub-directory in the specified output directory which contains saved models, 
all command line options (default and set by user) in an `args.txt` file as well as 
[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) logs. The scale factor is set with the
`--scale` option.

By default, the model is validated against randomly cropped images from the DIV2K validation set. If you'd rather
want the model to be evaluated against the full-sized images of the DIV2K validation set (= benchmark) after each 
epoch you need to set the `--benchmark` command line option. This however slows down training significantly and makes 
only sense for smaller models. Alternatively, you can benchmark saved models later with `bench.py` as described in 
the next section. 

For training models for higher scales (x3 or x4) it is recommended to re-use the weights of a corresponding model 
trained for a smaller scale. This can be done with the `--pretrained-model` command line option. For example,

    python train.py --dataset ./DIV2K --outdir ./output --profile wdsr-b-8 --scale 4 \ 
        --pretrained-model ./output/20181016-063620/models/epoch-294-psnr-34.5394.h5

trains a  WDSR-B baseline model with 8 residual blocks for scale x4 re-using the weights of the specified x2 model.
For a more detailed overview of available command line options and profiles please take a closer look at [`args.py`](args.py).

## Evaluation

An alternative to the `--benchmark` training option is to evaluate saved models with `bench.py` and then select the
model with the highest PSNR. For example

    python bench.py -i ./output/20181016-063620/models -o ./output/20181016-063620/bench.json
    
evaluates all models in directory `./output/20181016-063620/models` and writes the results as JSON data structure to
`./output/20181016-063620/bench.json`. The `bench.py` script also writes the best model in terms of PSNR to `stdout`
at the end of evaluation like:

    Best PSNR = 34.5394 for model ./output/20181016-063620/models/epoch-294-psnr-37.4630.h5 

The higher PSNR value in the model filename was generated during evaluation against smaller, randomly cropped images 
during training and must not be confused with the values generated by `bench.py`.
