# Single Image Super-Resolution with WDSR and EDSR

A [Keras](https://keras.io/)-based implementation of

- [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR), winner 
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge.
- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR), winner 
  of the [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) super-resolution challenge.

## Setup

This project requires Python 3.6 or higher.

1. Create a new [virtual environment](https://docs.python.org/3/tutorial/venv.html) or 
   [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) and activate that environment.

2. Run `pip install -r requirements-gpu.txt` if you have a GPU or `pip install -r requirements-cpu.txt` otherwise.


## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/13YjKmP5O8NK_E_dFlK-34Okn1IIM9c58). 
Each directory contains a model together with the training settings. All of them were trained with bicubic downscaled 
images from the DIV2K training set. Random crops and transformations were made as described in the EDSR paper. Model 
performance is measured in dB [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) on the DIV2K benchmark 
(images 801-900 of DIV2K validation set, RGB channels, without self-ensemble). See also section [Training](#training).

<table>
    <tr>
        <th>Model</th>
        <th>Scale</th>
        <th>Residual<br/>blocks </th>
        <th>Parameters</th>
        <th>PSNR</th>
        <th>Training</th>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1V4XHMFZo35yB_NTaD0dyw1_plS-78-Ju">wdsr-b-32-x2</a></td>
        <td>x2</td>
        <td>32</td>
        <td>3.55M</td>
        <td>34.80 dB</td>
        <td><a href="https://drive.google.com/open?id=1UgWCb7sSaKjDZDsZE93HhBEm4Rg7ofpa">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1ZTIz1YVXFTI2z3rvBfVuBSthJLJZivxC">wdsr-b-32-x4</a></td>
        <td>x4</td>
        <td>32</td>
        <td>3.56M</td>
        <td>29.17 dB</td>
        <td><a href="https://drive.google.com/open?id=1RhmgJkqZ86LEWfA7CAPfqBGhmNQ7Y7k7">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1Q2-fPMWm9EPGh4XEnfXKcxcSHuDik_3a">wdsr-b-16-x2</a></td>
        <td>x2</td>
        <td>16</td>
        <td>1.78M</td>
        <td>34.66 dB</td>
        <td><a href="https://drive.google.com/open?id=1iCTCzSd6bDr0h_J0bTRS3xB8SDyshHj-">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1xifqCrJeCypsMGzL-SWj7wzdNMCn35S-">wdsr-b-16-x4</a></td>
        <td>x4</td>
        <td>16</td>
        <td>1.79M</td>
        <td>29.04 dB</td>
        <td><a href="https://drive.google.com/open?id=1DzqDHiyy5xTbrwYKSU9hjRkNfoVAA7Vj">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1Vr_eLXnNA7H6zNWmEFKOBv4-xvOBt5iu">wdsr-b-8-x2</a><sup> 1)</sup></td>
        <td>x2</td>
        <td>8</td>
        <td>0.89M</td>
        <td>34.54 dB</td>
        <td><a href="https://drive.google.com/open?id=1VL4i4i1XuMy65wbq8fiWOOfMNziRqmdE">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1CSdinKy9E3B4dm-lp7O_W-MYXp0GoB9g">wdsr-b-8-x3</a></td>
        <td>x3</td>
        <td>8</td>
        <td>0.89M</td>
        <td>30.87 dB</td>
        <td><a href="https://drive.google.com/open?id=1B2w-ZSlD96RkCQ5C_JbQEDrdIMez7y3D">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1WCpIY9G-9fL9cTa3We9ry3hm-ePT58b_">wdsr-b-8-x4</a></td>
        <td>x4</td>
        <td>8</td>
        <td>0.90M</td>
        <td>28.91 dB</td>
        <td><a href="https://drive.google.com/open?id=1jgQfwGR_HVqVUjQqkvHCDhHowvTBmP5_">settings</a></td>
    </tr>
    <tr>
        <td>wdsr-a-16-x2</a><sup> 2)</sup></td>
        <td>x2</td>
        <td>16</td>
        <td>1.19M</td>
        <td>34.77 dB</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1tp7r_oUf8Ohd9q-ouGApS7qNtqg1IRLt">wdsr-a-8-x2</a></td>
        <td>x2</td>
        <td>8</td>
        <td>0.60M</td>
        <td>34.47 dB</td>
        <td><a href="https://drive.google.com/open?id=1hnL23k9_UYvGeAhY2nWOMM1rP2k-t8d-">settings</a></td>
    </tr>
    <tr>
        <td>edsr-16-x2</a><sup> 3)</sup></td>
        <td>x2</td>
        <td>16</td>
        <td>1.37M</td>
        <td>34.61 dB</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1ujCCDTJIheyGW-2wLU96tH13dGMEg84i">edsr-8-x2</a><sup> 4)</sup></td>
        <td>x2</td>
        <td>8</td>
        <td>0.78M</td>
        <td>34.41 dB</td>
        <td><a href="https://drive.google.com/open?id=1x8EjZxvTt0WO4zSdLDgBkKep3jYntrWc">settings</a></td>
    </tr>
</table>

<sup>1)</sup> This is the WDSR-B baseline with 8 residual blocks referenced on the [WDSR project page](https://github.com/JiahuiYu/wdsr_ntire2018). 
Measured PSNRs are identical.

<sup>2)</sup> This is the WDSR baseline with 16 residual blocks referenced on the [WDSR project page](https://github.com/JiahuiYu/wdsr_ntire2018).
PSNR is taken from there until a pre-trained model is available here.

<sup>3)</sup> This is the EDSR baseline referenced on the [EDSR project page](https://github.com/thstkdgus35/EDSR-PyTorch).
PSNR is taken from there until a pre-trained model is available here.

<sup>4)</sup> This is a smaller EDSR baseline model than that referenced on the [EDSR project page](https://github.com/thstkdgus35/EDSR-PyTorch). 
It has only 8 residual blocks instead of 16.

## Demo

The example in this section super-resolves images in directory [`./demo`](demo) with factor x4 using the downloaded 
[wdsr-b-32-x4]() model and writes the 
results to directory `./output`:

    python demo.py -i ./demo -o ./output --model=./wdsr-b-32-x4-psnr-29.1736.h5
    
The following figures compare the super-resolution results (SR) with the corresponding low-resolution (LR) and 
high-resolution (HR) images and an x4 resize with bicubic interpolation. The demo images were cropped from images in 
the DIV2K validation set. 

![0829](docs/demo-0829.png)

![0851](docs/demo-0851.png)

## DIV2K dataset

### Download

If you want to [train](#training) and [evaluate](#evaluation) models, you must download the 
[DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and extract the downloaded archives to a directory of your 
choice (`DIV2K` in the following example). The resulting directory structure should look like:
  
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

### Convert

Before the DIV2K images can be used they must be converted to numpy arrays and stored in a separate location. Conversion 
to numpy arrays dramatically reduces image loading times. Conversion can be done with the `convert.py` script: 

    python convert.py -i ./DIV2K -o ./DIV2K_BIN

In this example, converted images are written to the `DIV2K_BIN` directory. You'll later refer to this directory with the `--dataset` 
command line option. 

## Training

WDSR and EDSR models can be trained by running `train.py` with the command line options and profiles described in 
[`args.py`](args.py). For example, a WDSR-B baseline model with 8 residual blocks can be trained for scale x2 with

    python train.py --dataset ./DIV2K_BIN --outdir ./output --profile wdsr-b-8 --scale 2
    
The `--dataset` option sets the location of the DIV2K dataset and the `--output` option the output directory (defaults
to `./output`). Each training run creates a timestamped sub-directory in the specified output directory which contains 
saved models, all command line options (default and user-defined) in an `args.txt` file as well as 
[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) logs. The scale factor is set with the
`--scale` option. The downgrade operator can be set with the `--downgrade` option. It defaults to `bicubic` and can
be changed to `unknown`.

By default, the model is validated against randomly cropped images from the DIV2K validation set. If you'd rather
want to evaluate the model against the full-sized DIV2K validation images (= benchmark) after each epoch you need 
to set the `--benchmark` command line option. This however slows down training significantly and makes only sense 
for smaller models. Alternatively, you can benchmark saved models later with `bench.py` as described in the section
[Evaluation](#evaluation). 

To train models for higher scales (x3 or x4) it is recommended to re-use the weights of a model pre-trained for a 
smaller scale (x2). This can be done with the `--pretrained-model` option. For example,

    python train.py --dataset ./DIV2K_BIN --outdir ./output --profile wdsr-b-8 --scale 4 \ 
        --pretrained-model ./output/20181016-063620/models/epoch-294-psnr-34.5394.h5

trains a WDSR-B baseline model with 8 residual blocks for scale x4 re-using the weights of model 
`epoch-294-psnr-34.5394.h5`, a WDSR-B baseline model with the same number of residual blocks trained for scale x2. 

For a more detailed overview of available command line options and profiles please take a look at [`args.py`](args.py).

## Evaluation

An alternative to the `--benchmark` training option is to evaluate saved models with `bench.py` and then select the
model with the highest PSNR. For example,

    python bench.py -i ./output/20181016-063620/models -o bench.json
    
evaluates all models in directory `./output/20181016-063620/models` and writes the results to `bench.json`. This JSON
file maps model filenames to evaluation PSNR. The `bench.py` script also writes the best model in terms of PSNR to `stdout`
at the end of evaluation:

    Best PSNR = 34.5394 for model ./output/20181016-063620/models/epoch-294-psnr-37.4630.h5 

The higher PSNR value in the model filename must not be confused with the value generated by `bench.py`. The PSNR value 
in the filename was generated during training by validating against smaller, randomly cropped images which tends to yield
higher PSNR values.

## Other implementations

- [Official PyTorch implementation of the WDSR paper](https://github.com/JiahuiYu/wdsr_ntire2018) 
- [Official PyTorch implementation of the EDSR paper](https://github.com/thstkdgus35/EDSR-PyTorch) 
- [Official Torch implementation of the EDSR paper](https://github.com/LimBee/NTIRE2017)
- [Tensorflow implementation of the EDSR paper](https://github.com/jmiller656/EDSR-Tensorflow)

## Limitations

Code in this project requires the Keras Tensorflow backend.
