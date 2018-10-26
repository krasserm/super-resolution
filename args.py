import argparse


def int_range(s):
    try:
        fr, to = s.split('-')
        return range(int(fr), int(to) + 1)
    except Exception:
        raise argparse.ArgumentTypeError(f'invalid integer range: {s}')


def set_profile(args):

    # ----------------------------------
    #  WDSR-B profiles
    # ----------------------------------

    # WDSR-B baseline (as described in WDSR paper and implemented in https://github.com/JiahuiYu/wdsr_ntire2018)
    if args.profile == 'wdsr-b-8':
        args.model = 'wdsr-b'
        args.optimizer = 'adam-weightnorm'
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 8
        args.res_expansion = 6

    # WDSR-B baseline (modified to 16 residual blocks)
    elif args.profile == 'wdsr-b-16':
        args.model = 'wdsr-b'
        args.optimizer = 'adam-weightnorm'
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 16
        args.res_expansion = 6

    # WDSR-B baseline (modified to 32 residual blocks)
    elif args.profile == 'wdsr-b-32':
        args.model = 'wdsr-b'
        args.optimizer = 'adam-weightnorm'
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
        args.optimizer = 'adam-weightnorm'
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 8
        args.res_expansion = 4

    # WDSR-A baseline (modified to 16 residual blocks)
    elif args.profile == 'wdsr-a-16':
        args.model = 'wdsr-a'
        args.optimizer = 'adam-weightnorm'
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 16
        args.res_expansion = 4

    # WDSR-A baseline (modified to 32 residual blocks)
    elif args.profile == 'wdsr-a-32':
        args.model = 'wdsr-a'
        args.optimizer = 'adam-weightnorm'
        args.learning_rate = 1e-3
        args.num_filters = 32
        args.num_res_blocks = 32
        args.res_expansion = 4

    # ----------------------------------
    #  EDSR profiles
    # ----------------------------------

    # EDSR baseline (low-level)
    elif args.profile == 'edsr-8':
        args.model = 'edsr'
        args.optimizer = 'adam'
        args.learning_rate = 1e-4
        args.num_filters = 64
        args.num_res_blocks = 8

    # EDSR baseline (as described in EDSR paper)
    elif args.profile == 'edsr-16':
        args.model = 'edsr'
        args.optimizer = 'adam'
        args.learning_rate = 1e-4
        args.num_filters = 64
        args.num_res_blocks = 16

    # EDSR (as described in EDSR paper)
    elif args.profile == 'edsr':
        args.model = 'edsr'
        args.optimizer = 'adam'
        args.learning_rate = 1e-4
        args.num_filters = 256
        args.num_res_blocks = 32
        args.res_scaling = 0.1


parser = argparse.ArgumentParser(description='WDSR and EDSR training')

parser.add_argument('-p', '--profile', type=str,
                    choices=['wdsr-b-8', 'wdsr-b-16', 'wdsr-b-32',
                             'wdsr-a-8', 'wdsr-a-16', 'wdsr-a-32',
                             'edsr-8', 'edsr-16', 'edsr'],
                    help='model specific argument profiles')
parser.add_argument('-o', '--outdir', type=str, default='./output',
                    help='output directory')

# --------------
#  Dataset
# --------------

parser.add_argument('-d', '--dataset', type=str, default='./dataset',
                    help='path to DIV2K dataset with images stored as numpy arrays')
parser.add_argument('-s', '--scale', type=int, default=2, choices=[2, 3, 4],
                    help='super-resolution scale')
parser.add_argument('--downgrade', type=str, default='bicubic', choices=['bicubic', 'unknown'],
                    help='downgrade operation')
parser.add_argument('--training-images', type=int_range, default='1-800',
                    help='training image ids')
parser.add_argument('--validation-images', type=int_range, default='801-900',
                    help='validation image ids')
# --------------
#  Model
# --------------

parser.add_argument('-m', '--model', type=str, default='wdsr-b', choices=['edsr', 'wdsr-a', 'wdsr-b'],
                    help='model name')
parser.add_argument('--num-filters', type=int, default=32,
                    help='number of output filters in convolutions')
parser.add_argument('--num-res-blocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--res-expansion', type=float, default=6,
                    help='expansion factor r in WDSR models')
parser.add_argument('--res-scaling', type=float,
                    help='residual scaling factor')
# --------------
#  Training
# --------------

parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--steps-per-epoch', type=int, default=1000,
                    help='number of steps per epoch')
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
parser.add_argument('--optimizer', type=str, default='adam-weightnorm', choices=['adam', 'adam-weightnorm'],
                    help='optimizer to use')
parser.add_argument('--num-init-batches', type=int, default=0,
                    help='number of mini-batches for data-based weight initialization (adam-weightnorm optimizer only)')
parser.add_argument('--pretrained-model', type=str,
                    help='path to pre-trained model used for weight initialization')
parser.add_argument('--save-best-models-only', action='store_true',
                    help='save only models with improved validation psnr (overridden by --benchmark)')
parser.add_argument('--benchmark', action='store_true',
                    help='run DIV2K benchmark after each epoch and save best models only')
parser.add_argument('--num-workers', type=int, default=2,
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

