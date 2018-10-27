import os
import glob
import argparse
import numpy as np

from tqdm import tqdm
from util import load_image


def images_to_numpy(input_path, output_path):
    """
    Converts DIV2K images to numpy arrays preserving the directory structure.

    :param input_path: path to DIV2K images
    :param output_path: path to DIV2K numpy array to be generated
    """

    img_paths = glob.glob(os.path.join(input_path, '**', '*.png'), recursive=True)
    for img_path in tqdm(img_paths):
        img_dir, img_file = os.path.split(img_path)
        img_id, img_ext = os.path.splitext(img_file)

        rel_dir = os.path.relpath(img_dir, input_path)
        out_dir = os.path.join(output_path, rel_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        arr_path = os.path.join(out_dir, f'{img_id}.npy')

        img = load_image(img_path)
        np.save(arr_path, np.array(img, dtype='uint8'))


def main(args):
    images_to_numpy(args.indir, args.outdir)


def parser():
    parser = argparse.ArgumentParser(description='DIV2K image to numpy converter')

    parser.add_argument('-i', '--indir', type=str, required=True,
                        help='path to DIV2K images dataset')
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help='directory where converted image files are stored')

    return parser


if __name__ == '__main__':
    main(parser().parse_args())
