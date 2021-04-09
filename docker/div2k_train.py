import os
import matplotlib.pyplot as plt

from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer

# %matplotlib inline

# Location of model weights (needed for demo)
weights_dir = '/data/weights/div2k'
os.makedirs(weights_dir, exist_ok=True)
weights_file = lambda filename: os.path.join(weights_dir, filename)

# Set up images
images_dir = '/data/images'
os.makedirs(images_dir, exist_ok=True)
caches_dir = '/data/caches'
os.makedirs(caches_dir, exist_ok=True)
div2k_train = DIV2K(
                    scale=4,
                    subset='train',
                    downgrade='bicubic',
                    images_dir=images_dir,
                    caches_dir=caches_dir,
)
div2k_valid = DIV2K(
                    scale=4,
                    subset='valid',
                    downgrade='bicubic',
                    images_dir=images_dir,
                    caches_dir=caches_dir,
)

# Do pre-training
check_dir = f'/data/ckpt/pre_generator'
os.makedirs(check_dir, exist_ok=True)
train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)
pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=check_dir)
pre_trainer.train(
                    train_ds,
                    valid_ds.take(1),
                    steps=1000000, 
                    #steps=1000, 
                    evaluate_every=1000, 
                    save_best_only=False
)
pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

# Do gan-training
gan_generator = generator()
gan_generator.load_weights(weights_file('pre_generator.h5'))
gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
gan_trainer.train(
                    train_ds,
                    steps=200000
                    #steps=100
)
gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))
