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

All models were trained with images 1-800 from the DIV2K training set using the specified downgrade operator. Random 
crops and transformations were made as described in the EDSR paper. Model performance is measured in dB [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) 
on the DIV2K benchmark (images 801-900 of DIV2K validation set, RGB channels, without self-ensemble). See also section 
[Training](#training).

### Baseline models

<table>
    <tr>
        <th>Model</th>
        <th>Scale</th>
        <th>Residual<br/>blocks</th>
        <th>Downgrade</th>
        <th>Parameters</th>
        <th>PSNR</th>
        <th>Training</th>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1WmuSpNamFSszQOafrno05o1nDN4QjMeq">wdsr-a-16-x2</a><sup> 1)</sup></td>
        <td>x2</td>
        <td>16</td>
        <td>bicubic</td>
        <td>1.19M</td>
        <td>34.68 dB</td>
        <!--<td><a href="https://drive.google.com/open?id=1Lih9k_LHKw6hk9zJ6HjgGf-Mvz6ecNcE">settings</a></td>-->
        <td><details><summary>command</summary><pre>python train.py --profile wdsr-a-16 <br>--scale 2</pre></details></td>        
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1oATD-iXlQpcE2mIIEd4-9FOk2Xt5N8oX">edsr-16-x2</a><sup> 2)</sup></td>
        <td>x2</td>
        <td>16</td>
        <td>bicubic</td>
        <td>1.37M</td>
        <td>34.64 dB</td>
        <!--<td><a href="https://drive.google.com/open?id=1FgUfk7UK0f6y4eAcdAOWroREnVsRfV8Z">settings</a></td>-->
        <td><details><summary>command</summary><pre>python train.py --profile edsr-16 <br>--scale 2</pre></details></td>        
    </tr>
</table>

### Experimental models

<table>
    <tr>
        <th>Model</th>
        <th>Scale</th>
        <th>Residual<br/>blocks</th>
        <th>Downgrade</th>
        <th>Parameters</th>
        <th>PSNR</th>
        <th>Training</th>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1V4XHMFZo35yB_NTaD0dyw1_plS-78-Ju">wdsr-a-32-x2</a></td>
        <td>x2</td>
        <td>32</td>
        <td>bicubic</td>
        <td>3.55M<sup> 3)</sup></td>
        <td>34.80 dB</td>
        <!--<td><a href="https://drive.google.com/open?id=1UgWCb7sSaKjDZDsZE93HhBEm4Rg7ofpa">settings</a></td>-->
        <td><details><summary>command</summary><pre>python train.py --profile wdsr-a-32 <br>--scale 2 --res-expansion 6</pre></details></td>        
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1ZTIz1YVXFTI2z3rvBfVuBSthJLJZivxC">wdsr-a-32-x4</a></td>
        <td>x4</td>
        <td>32</td>
        <td>bicubic</td>
        <td>3.56M<sup> 3)</sup></td>
        <td>29.17 dB</td>
        <!--<td><a href="https://drive.google.com/open?id=1RhmgJkqZ86LEWfA7CAPfqBGhmNQ7Y7k7">settings</a></td>-->
        <td><details><summary>command</summary><pre>python train.py --profile wdsr-a-32 <br>--scale 4 --res-expansion 6 <br>--pretrained-model wdsr-a-32-x2-psnr-34.8033.h5</pre></details></td>        
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1XatcgjJM1s7BD_nHr8ApnyMhTEozY8SI">wdsr-a-32-x2-q90</a></td>
        <td>x2</td>
        <td>32</td>
        <td>bicubic + JPEG (90)<sup> 4)</sup></td>
        <td>3.55M<sup> 3)</sup></td>
        <td>32.12 dB</td>
        <!--<td><a href="https://drive.google.com/open?id=1mhPWkeUhu6d8LAzoFnI85lobcZSfaui0">settings</a></td>-->
        <td><details><summary>command</summary><pre>python train.py --profile wdsr-a-32 <br>--scale 2 --res-expansion 6 <br>--downgrade bicubic_jpeg_90 </pre></details></td>        
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1YBAPemerjTEA2OpQUP2o9iD9BZXdloJQ">wdsr-a-32-x4-q90</a></td>
        <td>x4</td>
        <td>32</td>
        <td>bicubic + JPEG (90)<sup> 4)</sup></td>
        <td>3.56M<sup> 3)</sup></td>
        <td>27.63 dB</td>
        <!--<td><a href="https://drive.google.com/open?id=1J4DZv_OCFDrtD82EidYqhMismGFQIhD6">settings</a></td>-->
        <td><details><summary>command</summary><pre>python train.py --profile wdsr-a-32 <br>--scale 4 --res-expansion 6 <br>--downgrade bicubic_jpeg_90 <br>--pretrained-model wdsr-a-32-x2-q90-psnr-32.1198.h5</pre></details></td>        
    </tr>
    <tr>
        <td><a href="https://drive.google.com/open?id=1_u80R7PA4HauacDw974-hBNfCcnQ9Fah">wdsr-b-32-x2</a></td>
        <td>x2</td>
        <td>32</td>
        <td>bicubic</td>
        <td>0.59M</td>
        <td>34.63 dB</td>
        <!--<td><a href="https://drive.google.com/open?id=1z-XMfUdW1WHHYHPQVILmRcoxbn1nwsva">settings</a></td>-->
        <td><details><summary>command</summary><pre>python train.py --profile wdsr-b-32 <br>--scale 2</pre></details></td>        
    </tr>
</table>


<sup>1)</sup> WDSR baseline(s), see also [WDSR project page](https://github.com/JiahuiYu/wdsr_ntire2018).  
<sup>2)</sup> EDSR baseline, see also [EDSR project page](https://github.com/thstkdgus35/EDSR-PyTorch).   
<sup>3)</sup> For experimental WDSR-A models, an expansion ratio of 6 was used, increasing the number of parameters
compared to an expansion ratio of 4. Please note that the default expansion ratio is 4 when using one the of the 
`wdsr-a-*` profiles with the `--profile` command line option for training. The default expansion ratio for WDSR-B 
models is 6.  
<sup>4)</sup> JPEG compression with quality `90` in addition to bicubic downscale. See also section 
[JPEG compression](#jpeg-compression).

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

If you want to [train](#training) and [evaluate](#evaluation) models, download the 
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

In this example, converted images are written to the `DIV2K_BIN` directory. By default, training and evaluation scripts 
read from this directory which can be overriden with the `--dataset` command line option. 

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

    python bench.py --dataset ./DIV2K_BIN -i ./output/20181016-063620/models -o bench.json
    
evaluates all models in directory `./output/20181016-063620/models` and writes the results to `bench.json`. This JSON
file maps model filenames to evaluation PSNR. The `bench.py` script also writes the model with the best PSNR to `stdout`
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

### WDSR

- [Official PyTorch implementation](https://github.com/JiahuiYu/wdsr_ntire2018) 
- [Official Tensorflow implementation](https://github.com/ychfan/tf_estimator_barebone/blob/master/docs/super_resolution.md) 

### EDSR

- [Official PyTorch implementation](https://github.com/thstkdgus35/EDSR-PyTorch) 
- [Official Torch implementation](https://github.com/LimBee/NTIRE2017)
- [Tensorflow implementation](https://github.com/jmiller656/EDSR-Tensorflow) by [Josh Miller](https://github.com/jmiller656).

## Limitations

Code in this project requires the Keras Tensorflow backend.
