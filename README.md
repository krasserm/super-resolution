![Travis CI](https://travis-ci.com/krasserm/wdsr.svg?branch=master)

# Single Image Super-Resolution with WDSR and EDSR

A [Keras](https://keras.io/)-based implementation of

- [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR), winner 
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge.
- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR), winner 
  of the [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) super-resolution challenge.

## Setup

Create a new [Conda](https://conda.io) environment with 

    conda env create -f environment-gpu.yml
    
if you have a GPU<sup>*)</sup>. A CPU-only environment can be created with

    conda env create -f environment-cpu.yml

Activate the environment with

    source activate wdsr
    
<sup>*)</sup> It is assumed that appropriate [CUDA](https://developer.nvidia.com/cuda-toolkit) and 
[cuDNN](https://developer.nvidia.com/cudnn) versions for the current [tensorflow-gpu](https://www.tensorflow.org/install/gpu) 
version are already installed on your system.

## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/13YjKmP5O8NK_E_dFlK-34Okn1IIM9c58). 
Each directory contains a model together with the training settings. All of them were trained with images 1-800 from 
the DIV2K training set using the specified downgrade operator. Random crops and transformations were made as described 
in the EDSR paper. Model performance is measured in dB [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) 
on the DIV2K benchmark (images 801-900 of DIV2K validation set, RGB channels, without self-ensemble). See also section 
[Training](#training).

### Experimental models

<table>
    <tr>
        <th>Model</th>
        <th>Scale</th>
        <th>Residual<br/>blocks</th>
        <th>Downgrade</th>
        <th>Parameters<sup> 1)</sup></th>
        <th>PSNR</th>
        <th>Training</th>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1V4XHMFZo35yB_NTaD0dyw1_plS-78-Ju">wdsr-a-32-x2</a></td>
        <td>x2</td>
        <td>32</td>
        <td>bicubic</td>
        <td>3.55M</td>
        <td>34.80 dB</td>
        <td><a href="https://drive.google.com/open?id=1UgWCb7sSaKjDZDsZE93HhBEm4Rg7ofpa">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1ZTIz1YVXFTI2z3rvBfVuBSthJLJZivxC">wdsr-a-32-x4</a></td>
        <td>x4</td>
        <td>32</td>
        <td>bicubic</td>
        <td>3.56M</td>
        <td>29.17 dB</td>
        <td><a href="https://drive.google.com/open?id=1RhmgJkqZ86LEWfA7CAPfqBGhmNQ7Y7k7">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1XatcgjJM1s7BD_nHr8ApnyMhTEozY8SI">wdsr-a-32-x2-q90</a></td>
        <td>x2</td>
        <td>32</td>
        <td>bicubic + JPEG (90)<sup> 2)</sup></td>
        <td>3.55M</td>
        <td>34.80 dB</td>
        <td><a href="https://drive.google.com/open?id=1mhPWkeUhu6d8LAzoFnI85lobcZSfaui0">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1YBAPemerjTEA2OpQUP2o9iD9BZXdloJQ">wdsr-a-32-x4-q90</a></td>
        <td>x4</td>
        <td>32</td>
        <td>bicubic + JPEG (90)<sup> 2)</sup></td>
        <td>3.56M</td>
        <td>29.17 dB</td>
        <td><a href="https://drive.google.com/open?id=1J4DZv_OCFDrtD82EidYqhMismGFQIhD6">settings</a></td>
    </tr>
</table>

### Baseline models

<table>
    <tr>
        <th>Model</th>
        <th>Scale</th>
        <th>Residual<br/>blocks</th>
        <th>Downgrade</th>
        <th>Parameters<sup> 1)</sup></th>
        <th>PSNR</th>
        <th>Training</th>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1Vr_eLXnNA7H6zNWmEFKOBv4-xvOBt5iu">wdsr-a-8-x2</a><sup> 3)</sup></td>
        <td>x2</td>
        <td>8</td>
        <td>bicubic</td>
        <td>0.89M</td>
        <td>34.54 dB</td>
        <td><a href="https://drive.google.com/open?id=1VL4i4i1XuMy65wbq8fiWOOfMNziRqmdE">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1CSdinKy9E3B4dm-lp7O_W-MYXp0GoB9g">wdsr-a-8-x3</a></td>
        <td>x3</td>
        <td>8</td>
        <td>bicubic</td>
        <td>0.89M</td>
        <td>30.87 dB</td>
        <td><a href="https://drive.google.com/open?id=1B2w-ZSlD96RkCQ5C_JbQEDrdIMez7y3D">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1WCpIY9G-9fL9cTa3We9ry3hm-ePT58b_">wdsr-a-8-x4</a></td>
        <td>x4</td>
        <td>8</td>
        <td>bicubic</td>
        <td>0.90M</td>
        <td>28.91 dB</td>
        <td><a href="https://drive.google.com/open?id=1jgQfwGR_HVqVUjQqkvHCDhHowvTBmP5_">settings</a></td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1ujCCDTJIheyGW-2wLU96tH13dGMEg84i">edsr-8-x2</a><sup> 4)</sup></td>
        <td>x2</td>
        <td>8</td>
        <td>bicubic</td>
        <td>0.78M</td>
        <td>34.41 dB</td>
        <td><a href="https://drive.google.com/open?id=1x8EjZxvTt0WO4zSdLDgBkKep3jYntrWc">settings</a></td>
    </tr>
</table>

<sup>1)</sup> For WDSR-A models an expansion ratio of 6 was used, instead of 4, without a decrease in performance. Please 
note that the default expansion ratio is 4 when using one the of the `wdsr-a-*` profiles with the `--profile` command 
line option.

<sup>2)</sup> JPEG compression with quality `90` in addition to bicubic downscale. See also section 
[JPEG compression](#jpeg-compression).

<sup>3)</sup> WDSR-A baseline with 8 residual blocks referenced on the [WDSR project page](https://github.com/JiahuiYu/wdsr_ntire2018). 
Measured PSNRs are identical.

<sup>3)</sup> Smaller EDSR baseline model than that referenced on the [EDSR project page](https://github.com/thstkdgus35/EDSR-PyTorch)
(has only 8 residual blocks instead of 16).

## Demo

First, download the [wdsr-a-32-x4](https://drive.google.com/open?id=1ZTIz1YVXFTI2z3rvBfVuBSthJLJZivxC) model. Assuming 
that the path to the downloaded model is `~/Downloads/wdsr-a-32-x4-psnr-29.1736.h5`, the following command super-resolves
images in directory [`./demo`](demo) with factor x4 and writes the results to directory `./output`:

    python demo.py -i ./demo -o ./output --model ~/Downloads/wdsr-a-32-x4-psnr-29.1736.h5
    
Below are figures that compare the super-resolution results (SR) with the corresponding low-resolution (LR) and 
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

    python convert.py -i ./DIV2K -o ./DIV2K_BIN numpy

In this example, converted images are written to the `DIV2K_BIN` directory. You'll later refer to this directory with the 
`--dataset` command line option. 

### JPEG compression

There is experimental support for adding JPEG compression artifacts to LR images and training with these images. The 
following commands convert bicubic downscaled DIV2K training and validation images to JPEG images with quality `90`:

    python convert.py -i ./DIV2K/DIV2K_train_LR_bicubic \
                      -o ./DIV2K/DIV2K_train_LR_bicubic_jpeg_90 \
                       --jpeg-quality 90 jpeg

    python convert.py -i ./DIV2K/DIV2K_valid_LR_bicubic \
                      -o ./DIV2K/DIV2K_valid_LR_bicubic_jpeg_90 \
                       --jpeg-quality 90 jpeg

After having converted these JPEG images to numpy arrays, as described in the previous section, models can be trained
with the `--downgrade bicubic_jpeg_90` option to additionally learn to recover from JPEG compression artifacts.
 
## Training

WDSR and EDSR models can be trained by running `train.py` with the command line options and profiles described in 
[`train.py`](train.py). For example, a WDSR-A baseline model with 8 residual blocks can be trained for scale x2 with

    python train.py --dataset ./DIV2K_BIN --outdir ./output --profile wdsr-a-8 --scale 2
    
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

    python train.py --dataset ./DIV2K_BIN --outdir ./output --profile wdsr-a-8 --scale 4 \ 
        --pretrained-model ./output/20181016-063620/models/epoch-294-psnr-34.5394.h5

trains a WDSR-A baseline model with 8 residual blocks for scale x4 re-using the weights of model 
`epoch-294-psnr-34.5394.h5`, a WDSR-A baseline model with the same number of residual blocks trained for scale x2. 

For a more detailed overview of available command line options and profiles please take a look at [`train.py`](train.py).

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

## Tests

The test suite can be run with 

    pytest tests
    
## Weight normalization

WDSR models are trained with [weight normalization](https://arxiv.org/abs/1602.07868). This branch uses a 
[modified Adam optimizer](https://github.com/krasserm/wdsr/blob/master/optimizer/weightnorm.py). Branch 
[wip-conv2d-weight-norm](https://github.com/krasserm/wdsr/tree/wip-conv2d-weight-norm) instead uses a specialized 
[`Conv2DWeightNorm`](https://github.com/krasserm/wdsr/blob/wip-conv2d-weight-norm/layer.py) layer and a default Adam 
optimizer (experimental work inspired by the official [WDSR Tensorflow](https://github.com/ychfan/tf_estimator_barebone/blob/master/models/wdsr.py) 
port). Current plan is to replace this layer with a default `Conv2D` layer and a [Tensorflow WeightNorm wrapper](https://github.com/tensorflow/tensorflow/pull/21276)
when the wrapper is officially available in a Tensorflow release.
    
## Other implementations

- [Official PyTorch implementation of the WDSR paper](https://github.com/JiahuiYu/wdsr_ntire2018) 
- [Official PyTorch implementation of the EDSR paper](https://github.com/thstkdgus35/EDSR-PyTorch) 
- [Official Torch implementation of the EDSR paper](https://github.com/LimBee/NTIRE2017)
- [Tensorflow implementation of the EDSR paper](https://github.com/jmiller656/EDSR-Tensorflow)

## Limitations

Code in this project requires the Keras Tensorflow backend.
