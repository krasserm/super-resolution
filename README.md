![Travis CI](https://travis-ci.com/krasserm/super-resolution.svg?branch=master)

# Single Image Super-Resolution with EDSR, WDSR and SRGAN

A [Tensorflow 2.x](https://www.tensorflow.org/beta) based implementation of

- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR), winner 
  of the [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) super-resolution challenge.
- [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR), winner 
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge (realistic tracks).
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGAN).

This is a complete re-write of the old Keras/Tensorflow 1.x based implementation available [here](https://github.com/krasserm/super-resolution/tree/previous).
Some parts are still work in progress but you can already train models as described in the papers via a high-level training 
API. Furthermore, you can also [fine-tune](#srgan-for-fine-tuning-edsr-and-wdsr-models) EDSR and WDSR models in an SRGAN 
context. [Training](#training) and [usage](#getting-started) examples are given in the notebooks

- [example-edsr.ipynb](example-edsr.ipynb)
- [example-wdsr.ipynb](example-wdsr.ipynb)
- [example-srgan.ipynb](example-srgan.ipynb) 

A `DIV2K` [data provider](#div2k-dataset) automatically downloads [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
training and validation images of given scale (2, 3, 4 or 8) and downgrade operator ("bicubic", "unknown", "mild" or 
"difficult"). 

**Important:** if you want to evaluate the pre-trained models with a dataset other than DIV2K please read 
[this comment](https://github.com/krasserm/super-resolution/issues/19#issuecomment-586114933) (and replies) first.  

## Environment setup

Create a new [conda](https://conda.io) environment with

    conda env create -f environment.yml
    
and activate it with

    conda activate sisr

## Introduction

You can find an introduction to single-image super-resolution in [this article](https://krasserm.github.io/2019/09/04/super-resolution/). 
It also demonstrates how EDSR and WDSR models can be fine-tuned with SRGAN (see also [this section](#srgan-for-fine-tuning-edsr-and-wdsr-models)).

## Getting started 

Examples in this section require following pre-trained weights for running (see also example notebooks):  

### Pre-trained weights

- [weights-edsr-16-x4.tar.gz](https://martin-krasser.de/sisr/weights-edsr-16-x4.tar.gz) 
    - EDSR x4 baseline as described in the EDSR paper: 16 residual blocks, 64 filters, 1.52M parameters. 
    - PSNR on DIV2K validation set = 28.89 dB (images 801 - 900, 6 + 4 pixel border included).
- [weights-wdsr-b-32-x4.tar.gz](https://martin-krasser.de/sisr/weights-wdsr-b-32-x4.tar.gz) 
    - WDSR B x4 custom model: 32 residual blocks, 32 filters, expansion factor 6, 0.62M parameters. 
    - PSNR on DIV2K validation set = 28.91 dB (images 801 - 900, 6 + 4 pixel border included).
- [weights-srgan.tar.gz](https://martin-krasser.de/sisr/weights-srgan.tar.gz) 
    - SRGAN as described in the SRGAN paper: 1.55M parameters, trained with VGG54 content loss.
    
After download, extract them in the root folder of the project with

    tar xvfz weights-<...>.tar.gz

### EDSR

```python
from model import resolve_single
from model.edsr import edsr

from utils import load_image, plot_sample

model = edsr(scale=4, num_res_blocks=16)
model.load_weights('weights/edsr-16-x4/weights.h5')

lr = load_image('demo/0851x4-crop.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
```

![result-edsr](docs/images/result-edsr.png)

### WDSR

```python
from model.wdsr import wdsr_b

model = wdsr_b(scale=4, num_res_blocks=32)
model.load_weights('weights/wdsr-b-32-x4/weights.h5')

lr = load_image('demo/0829x4-crop.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
```

![result-wdsr](docs/images/result-wdsr.png)

Weight normalization in WDSR models is implemented with the new `WeightNormalization` layer wrapper of 
[Tensorflow Addons](https://github.com/tensorflow/addons). In its latest version, this wrapper seems to 
corrupt weights when running `model.predict(...)`. A workaround is to set `model.run_eagerly = True` or 
compile the model with `model.compile(loss='mae')` in advance. This issue doesn't arise when calling the
model directly with `model(...)` though. To be further investigated ... 

### SRGAN

```python
from model.srgan import generator

model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

lr = load_image('demo/0869x4-crop.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
```

![result-srgan](docs/images/result-srgan.png)

## DIV2K dataset

For training and validation on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) images, applications should use the 
provided `DIV2K` data loader. It automatically downloads DIV2K images to `.div2k` directory and converts them to a 
different format for faster loading.

### Training dataset

```python
from data import DIV2K

train_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='train')      # Training dataset are images 001 - 800
                     
# Create a tf.data.Dataset          
train_ds = train_loader.dataset(batch_size=16,         # batch size as described in the EDSR and WDSR papers
                                random_transform=True, # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)     # repeat iterating over training images indefinitely

# Iterate over LR/HR image pairs                                
for lr, hr in train_ds:
    # .... 
```

Crop size in HR images is 96x96. 

### Validation dataset

```python
from data import DIV2K

valid_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='valid')      # Validation dataset are images 801 - 900
                     
# Create a tf.data.Dataset          
valid_ds = valid_loader.dataset(batch_size=1,           # use batch size of 1 as DIV2K images have different size
                                random_transform=False, # use DIV2K images in original size 
                                repeat_count=1)         # 1 epoch
                                
# Iterate over LR/HR image pairs                                
for lr, hr in valid_ds:
    # ....                                 
```

## Training 

The following training examples use the [training and validation datasets](#div2k-dataset) described earlier. The high-level 
training API is designed around *steps* (= minibatch updates) rather than *epochs* to better match the descriptions in the 
papers.

## EDSR

```python
from model.edsr import edsr
from train import EdsrTrainer

# Create a training context for an EDSR x4 model with 16 
# residual blocks.
trainer = EdsrTrainer(model=edsr(scale=4, num_res_blocks=16), 
                      checkpoint_dir=f'.ckpt/edsr-16-x4')
                      
# Train EDSR model for 300,000 steps and evaluate model
# every 1000 steps on the first 10 images of the DIV2K
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.
trainer.train(train_ds,
              valid_ds.take(10),
              steps=300000, 
              evaluate_every=1000, 
              save_best_only=True)
              
# Restore from checkpoint with highest PSNR.
trainer.restore()

# Evaluate model on full validation set.
psnr = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}')

# Save weights to separate location.
trainer.model.save_weights('weights/edsr-16-x4/weights.h5')                                    
```

Interrupting training and restarting it again resumes from the latest saved checkpoint. The trained Keras model can be
accessed with `trainer.model`.

## WDSR

```python
from model.wdsr import wdsr_b
from train import WdsrTrainer

# Create a training context for a WDSR B x4 model with 32 
# residual blocks.
trainer = WdsrTrainer(model=wdsr_b(scale=4, num_res_blocks=32), 
                      checkpoint_dir=f'.ckpt/wdsr-b-8-x4')

# Train WDSR B model for 300,000 steps and evaluate model
# every 1000 steps on the first 10 images of the DIV2K
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.
trainer.train(train_ds,
              valid_ds.take(10),
              steps=300000, 
              evaluate_every=1000, 
              save_best_only=True)

# Restore from checkpoint with highest PSNR.
trainer.restore()

# Evaluate model on full validation set.
psnr = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}')

# Save weights to separate location.
trainer.model.save_weights('weights/wdsr-b-32-x4/weights.h5')
```

## SRGAN

### Generator pre-training

```python
from model.srgan import generator
from train import SrganGeneratorTrainer

# Create a training context for the generator (SRResNet) alone.
pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')

# Pre-train the generator with 1,000,000 steps (100,000 works fine too). 
pre_trainer.train(train_ds, valid_ds.take(10), steps=1000000, evaluate_every=1000)

# Save weights of pre-trained generator (needed for fine-tuning with GAN).
pre_trainer.model.save_weights('weights/srgan/pre_generator.h5')
```

### Generator fine-tuning (GAN)

```python
from model.srgan import generator, discriminator
from train import SrganTrainer

# Create a new generator and init it with pre-trained weights.
gan_generator = generator()
gan_generator.load_weights('weights/srgan/pre_generator.h5')

# Create a training context for the GAN (generator + discriminator).
gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())

# Train the GAN with 200,000 steps.
gan_trainer.train(train_ds, steps=200000)

# Save weights of generator and discriminator.
gan_trainer.generator.save_weights('weights/srgan/gan_generator.h5')
gan_trainer.discriminator.save_weights('weights/srgan/gan_discriminator.h5')
```

## SRGAN for fine-tuning EDSR and WDSR models

It is also possible to fine-tune EDSR and WDSR x4 models with SRGAN. They can be used as drop-in replacement for the
original SRGAN generator. More details in [this article](https://krasserm.github.io/2019/09/04/super-resolution/).

```python
# Create EDSR generator and init with pre-trained weights
generator = edsr(scale=4, num_res_blocks=16)
generator.load_weights('weights/edsr-16-x4/weights.h5')

# Fine-tune EDSR model via SRGAN training.
gan_trainer = SrganTrainer(generator=generator, discriminator=discriminator())
gan_trainer.train(train_ds, steps=200000)
```

```python
# Create WDSR B generator and init with pre-trained weights
generator = wdsr_b(scale=4, num_res_blocks=32)
generator.load_weights('weights/wdsr-b-16-32/weights.h5')

# Fine-tune WDSR B  model via SRGAN training.
gan_trainer = SrganTrainer(generator=generator, discriminator=discriminator())
gan_trainer.train(train_ds, steps=200000)
```
