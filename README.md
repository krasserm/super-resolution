# Single image super-resolution for Keras

A Keras-based implementation of

- [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR). Winner 
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge.
- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR). Winner 
  of the [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) super-resolution challenge.

## Training examples

    #Train WDSR-B x2 baseline (8 residual blocks)
    python3 train.py --dataset=<path-to-div2k> --outdir=<path-to-outdir> --profile=baseline-wdsr-b \
    --scale=2 --learning-rate=1e-3 --benchmark

    #Train WDSR-A x2 baseline (8 residual blocks)
    python3 train.py --dataset=<path-to-div2k> --outdir=<path-to-outdir> --profile=baseline-wdsr-a \
        --scale=2 --learning-rate=1e-3 --benchmark

    #Train EDSR x2 baseline (8 residual blocks)
    python3 train.py --dataset=<path-to-div2k> --outdir=<path-to-outdir> --profile=baseline-edsr \
        --scale=2 --learning-rate=1e-4 --benchmark

Detailed documentation and pre-trained models coming soon ...
