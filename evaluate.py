import os
import glob
import json
import logging
import argparse

from data import fullsize_sequence, DOWNGRADES
from model import load_model
from util import reset_session

logger = logging.getLogger(__name__)


def model_paths(input_dir):
    path_pattern = os.path.join(input_dir, '**', '*.h5')
    paths = glob.glob(path_pattern, recursive=True)
    paths.sort()
    return paths


def select_best_psnr(psnr_dict):
    best_psnr = 0.0
    best_model = None

    for model, psnr in psnr_dict.items():
        if psnr > best_psnr:
            best_psnr = psnr
            best_model = model

    return best_psnr, best_model


def evaluate_model(model_path, generator):
    """Evaluate model against DIV2K validation set and return PSNR"""
    logger.info('Load model %s', model_path)
    model = load_model(model_path)

    logger.info('Evaluate model %s', model_path)
    return model.evaluate_generator(generator, steps=100, verbose=1)[1]


def main(args):
    """
    Evaluate all models in a user-defined directory against the DIV2K validation set.

    The results are written to a user-defined JSON file. All models in the input
    directory must have been trained for the same downgrade operator (bicubic or
    unknown) and the same scale (2, 3 or 4).
    """

    mps = model_paths(args.indir)

    if mps:
        generator = fullsize_sequence(args.dataset, scale=args.scale, subset='valid', downgrade=args.downgrade)
        psnr_dict = {}
        for mp in mps:
            reset_session(args.gpu_memory_fraction)
            psnr = evaluate_model(mp, generator)
            logger.info('PSNR = %.4f for model %s', psnr, mp)
            psnr_dict[mp] = psnr

        logger.info('Write results to %s', args.outfile)
        with open(args.outfile, 'w') as f:
            json.dump(psnr_dict, f)

        best_psnr, best_model = select_best_psnr(psnr_dict)
        logger.info('Best PSNR = %.4f for model %s', best_psnr, best_model)
    else:
        logger.warning('No models found in %s', args.indir)


def parser():
    parser = argparse.ArgumentParser(description='Evaluation against DIV2K validation set')

    parser.add_argument('-d', '--dataset', type=str, default='./DIV2K_BIN',
                        help='path to DIV2K dataset with images stored as numpy arrays')
    parser.add_argument('-i', '--indir', type=str,
                        help='path to models directory')
    parser.add_argument('-o', '--outfile', type=str, default='./eval.json',
                        help='output JSON file')
    parser.add_argument('-s', '--scale', type=int, default=2, choices=[2, 3, 4],
                        help='super-resolution scale')
    parser.add_argument('--downgrade', type=str, default='bicubic', choices=DOWNGRADES,
                        help='downgrade operation')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
                        help='fraction of GPU memory to allocate')

    return parser

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    main(parser().parse_args())
