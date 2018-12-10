import os
import logging
import datetime
import itertools
import argparse
import numpy as np
import tensorflow as tf

from callback import learning_rate, model_checkpoint_after, tensor_board
from data import cropped_sequence, fullsize_sequence, DOWNGRADES
from model import copy_weights, edsr, wdsr, srgan
from optimizer import weightnorm as wn
from util import init_session

from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam

logger = logging.getLogger(__name__)


def create_train_workspace(path):
    train_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_dir = os.path.join(path, train_dir)
    models_dir = os.path.join(train_dir, 'models')
    os.makedirs(train_dir, exist_ok=True)
    os.mkdir(models_dir)
    return train_dir, models_dir


def write_args(path, args):
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for k, v in sorted(args.__dict__.items()):
            f.write(f'{k}={v}\n')


def model_weightnorm_init(model, generator, num_batches):
    lr_batches = [lr_batch for lr_batch, _ in itertools.islice(generator, num_batches)]
    wn.data_based_init(model, np.concatenate(lr_batches, axis=0))


def mae(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return mean_absolute_error(hr, sr)


def psnr(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return tf.image.psnr(hr, sr, max_val=255)


def _crop_hr_in_training(hr, sr):
    """
    Remove margin of size scale*2 from hr in training phase.

    The margin is computed from size difference of hr and sr
    so that no explicit scale parameter is needed. This is only
    needed for WDSR models.
    """

    margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2

    # crop only if margin > 0
    hr_crop = tf.cond(tf.equal(margin, 0),
                      lambda: hr,
                      lambda: hr[:, margin:-margin, margin:-margin, :])

    hr = K.in_train_phase(hr_crop, hr)
    hr.uses_learning_phase = True
    return hr, sr


def _load_model(path):
    return load_model(path, custom_objects={**_custom_objects, **_custom_objects_backwards_compat})


_custom_objects = {
    'tf': tf,
    'AdamWithWeightnorm': wn.AdamWithWeightnorm,
    'mae': mae,
    'psnr': psnr
}

_custom_objects_backwards_compat = {
    'mae_scale_2': mae,
    'mae_scale_3': mae,
    'mae_scale_4': mae,
    'psnr_scale_2': psnr,
    'psnr_scale_3': psnr,
    'psnr_scale_4': psnr
}


def main(args):
    train_dir, models_dir = create_train_workspace(args.outdir)
    write_args(train_dir, args)
    logger.info('Training workspace is %s', train_dir)

    training_generator = cropped_sequence(args.dataset, scale=args.scale, subset='train', downgrade=args.downgrade,
                                          image_ids=args.training_images, batch_size=args.batch_size)

    if args.benchmark:
        logger.info('Validation with full-size images from DIV2K validation set')
        validation_steps = len(args.validation_images)
        validation_generator = fullsize_sequence(args.dataset, scale=args.scale, subset='valid', downgrade=args.downgrade,
                                                 image_ids=args.validation_images)
    else:
        logger.info('Validation with randomly cropped images from DIV2K validation set')
        validation_steps = args.validation_steps
        validation_generator = cropped_sequence(args.dataset, scale=args.scale, subset='valid', downgrade=args.downgrade,
                                                image_ids=args.validation_images, batch_size=args.batch_size)

    if args.initial_epoch:
        logger.info('Resume training of model %s', args.pretrained_model)
        model = _load_model(args.pretrained_model)

    else:
        if args.model == "sr-resnet":
            #
            # Pre-training of SRResNet-based generator
            # (for usage in SRGAN)
            #
            loss = mean_squared_error
            model = srgan.generator(num_filters=args.num_filters,
                                    num_res_blocks=args.num_res_blocks)
        elif args.model == "edsr-gen":
            #
            # Pre-training of EDSR-based generator
            # (for usage in an SRGAN-like network)
            #
            loss = mean_squared_error
            model = edsr.edsr_generator(scale=args.scale,
                                        num_filters=args.num_filters,
                                        num_res_blocks=args.num_res_blocks)
        elif args.model == "edsr":
            loss = mean_absolute_error
            model = edsr.edsr(scale=args.scale,
                              num_filters=args.num_filters,
                              num_res_blocks=args.num_res_blocks,
                              res_block_scaling=args.res_scaling)
        else:
            loss = mae
            model_fn = wdsr.wdsr_b if args.model == 'wdsr-b' else wdsr.wdsr_a
            model = model_fn(scale=args.scale,
                             num_filters=args.num_filters,
                             num_res_blocks=args.num_res_blocks,
                             res_block_expansion = args.res_expansion,
                             res_block_scaling=args.res_scaling)

        if args.weightnorm:
            model.compile(optimizer=wn.AdamWithWeightnorm(lr=args.learning_rate), loss=loss, metrics=[psnr])
            if args.num_init_batches > 0:
                logger.info('Data-based initialization of weights with %d batches', args.num_init_batches)
                model_weightnorm_init(model, training_generator, args.num_init_batches)
        else:
            model.compile(optimizer=Adam(lr=args.learning_rate), loss=loss, metrics=[psnr])

        if args.pretrained_model:
            logger.info('Initialization with weights from pre-trained model %s', args.pretrained_model)
            copy_weights(from_model=_load_model(args.pretrained_model), to_model=model)

    if args.print_model_summary:
        model.summary()

    callbacks = [
        tensor_board(train_dir),
        learning_rate(step_size=args.learning_rate_step_size, decay=args.learning_rate_decay),
        model_checkpoint_after(args.save_models_after_epoch, models_dir, monitor=f'val_psnr',
                               save_best_only=args.save_best_models_only or args.benchmark)
    ]

    model.fit_generator(training_generator,
                        epochs=args.epochs,
                        initial_epoch=args.initial_epoch,
                        steps_per_epoch=args.iterations_per_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        use_multiprocessing=args.use_multiprocessing,
                        max_queue_size=args.max_queue_size,
                        workers=args.num_workers,
                        callbacks=callbacks)


def parser():
    parser = argparse.ArgumentParser(description='WDSR and EDSR training')

    parser.add_argument('-p', '--profile', type=str,
                        choices=['wdsr-b-8', 'wdsr-b-16', 'wdsr-b-32',
                                 'wdsr-a-8', 'wdsr-a-16', 'wdsr-a-32',
                                 'edsr-8', 'edsr-16', 'edsr', 'edsr-gen', 'sr-resnet'],
                        help='model specific argument profiles')
    parser.add_argument('-o', '--outdir', type=str, default='./output',
                        help='output directory')

    # --------------
    #  Dataset
    # --------------

    parser.add_argument('-d', '--dataset', type=str, default='./DIV2K_BIN',
                        help='path to DIV2K dataset with images stored as numpy arrays')
    parser.add_argument('-s', '--scale', type=int, default=2, choices=[2, 3, 4],
                        help='super-resolution scale')
    parser.add_argument('--downgrade', type=str, default='bicubic', choices=DOWNGRADES,
                        help='downgrade operation')
    parser.add_argument('--training-images', type=int_range, default='1-800',
                        help='training image ids')
    parser.add_argument('--validation-images', type=int_range, default='801-900',
                        help='validation image ids')
    # --------------
    #  Model
    # --------------

    parser.add_argument('-m', '--model', type=str, default='wdsr-b', choices=['edsr', 'edsr-gen', 'wdsr-a', 'wdsr-b', 'sr-resnet'],
                        help='model name')
    parser.add_argument('--num-filters', type=int, default=32,
                        help='number of filters')
    parser.add_argument('--num-res-blocks', type=int, default=8,
                        help='number of residual blocks')
    parser.add_argument('--res-expansion', type=int, default=4,
                        help='expansion factor r in WDSR models')
    parser.add_argument('--res-scaling', type=float,
                        help='residual scaling factor')
    # --------------
    #  Training
    # --------------

    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--iterations-per-epoch', type=int, default=1000,
                        help='number of update iterations per epoch')
    parser.add_argument('--validation-steps', type=int, default=100,
                        help='number of validation steps for validation')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='mini-batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--learning-rate-step-size', type=int, default=200,
                        help='learning rate step size in epochs')
    parser.add_argument('--learning-rate-decay', type=float, default=0.5,
                        help='learning rate decay at each step')
    parser.add_argument('--weightnorm', action='store_true',
                        help='train with weight normalization')
    parser.add_argument('--num-init-batches', type=int, default=0,
                        help='number of mini-batches for data-based weight initialization when using --weightnorm')
    parser.add_argument('--pretrained-model', type=str,
                        help='path to pre-trained model')
    parser.add_argument('--save-best-models-only', action='store_true',
                        help='save only models with improved validation psnr (overridden by --benchmark)')
    parser.add_argument('--save-models-after-epoch', type=int, default=0,
                        help='start saving models only after given epoch')
    parser.add_argument('--benchmark', action='store_true',
                        help='validate with full-size DIV2K images after each epoch and save best models only')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='resumes training of provided model if greater than 0')
    parser.add_argument('--use-multiprocessing', action='store_true',
                        help='use multi-processing for data loading')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of data loading workers')
    parser.add_argument('--max-queue-size', type=int, default=16,
                        help='maximum size for generator queue')
    parser.add_argument('--print-model-summary', action='store_true',
                        help='print model summary before training')

    # --------------
    #  Hardware
    # --------------

    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
                        help='fraction of GPU memory to allocate')

    return parser


def set_profile(args):

    # ----------------------------------
    #  WDSR-B profiles
    # ----------------------------------

    # WDSR-B baseline (as described in WDSR paper and implemented in https://github.com/JiahuiYu/wdsr_ntire2018)
    if args.profile == 'wdsr-b-8':
        args.model = 'wdsr-b'
        args.weightnorm = True
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 8
        args.res_expansion = 6

    # WDSR-B baseline (modified to 16 residual blocks)
    elif args.profile == 'wdsr-b-16':
        args.model = 'wdsr-b'
        args.weightnorm = True
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 16
        args.res_expansion = 6

    # WDSR-B baseline (modified to 32 residual blocks)
    elif args.profile == 'wdsr-b-32':
        args.model = 'wdsr-b'
        args.weightnorm = True
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 32
        args.res_expansion = 6

    # ----------------------------------
    #  WDSR-A profiles
    # ----------------------------------

    # WDSR-A baseline (as described in WDSR paper and implemented in https://github.com/JiahuiYu/wdsr_ntire2018)
    elif args.profile == 'wdsr-a-8':
        args.model = 'wdsr-a'
        args.weightnorm = True
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 8

    # WDSR-A baseline (modified to 16 residual blocks)
    elif args.profile == 'wdsr-a-16':
        args.model = 'wdsr-a'
        args.weightnorm = True
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 16

    # WDSR-A baseline (modified to 32 residual blocks)
    elif args.profile == 'wdsr-a-32':
        args.model = 'wdsr-a'
        args.weightnorm = True
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 32

    # ----------------------------------
    #  EDSR profiles
    # ----------------------------------

    # EDSR baseline (low-level)
    elif args.profile == 'edsr-8':
        args.model = 'edsr'
        args.learning_rate = 1e-4
        args.num_filters = 64
        args.num_res_blocks = 8

    # EDSR baseline (as described in EDSR paper)
    elif args.profile == 'edsr-16':
        args.model = 'edsr'
        args.learning_rate = 1e-4
        args.num_filters = 64
        args.num_res_blocks = 16

    # EDSR (as described in EDSR paper)
    elif args.profile == 'edsr':
        args.model = 'edsr'
        args.learning_rate = 1e-4
        args.num_filters = 256
        args.num_res_blocks = 32
        args.res_scaling = 0.1

    # -----------------------------------------
    #  Profiles for GAN generator pre-training
    # -----------------------------------------

    elif args.profile == 'edsr-gen':
        args.model = 'edsr-gen'
        args.learning_rate = 1e-4
        args.num_filters = 64


    elif args.profile == 'sr-resnet':
        args.model = 'sr-resnet'
        args.scale = 4
        args.learning_rate = 1e-4
        args.num_filters = 64
        args.num_res_blocks = 16


def int_range(s):
    try:
        fr, to = s.split('-')
        return range(int(fr), int(to) + 1)
    except Exception:
        raise argparse.ArgumentTypeError(f'invalid integer range: {s}')


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    args = parser().parse_args()
    set_profile(args)

    init_session(args.gpu_memory_fraction)
    main(args)
