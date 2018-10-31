import os
import glob
import argparse
import numpy as np

from tqdm import tqdm
from util import load_image


def convert(input_path, output_path, converter, extensions):
    """
    Converts DIV2K images using converter preserving the directory structure.

    :param input_path: path to DIV2K images
    :param output_path: path to DIV2K numpy array to be generated
    """

    img_paths = []

    for extension in extensions:
        img_paths_ext = glob.glob(os.path.join(input_path, '**', f'*.{extension}'), recursive=True)
        img_paths.extend(img_paths_ext)

    for img_path in tqdm(img_paths):
        img_dir, img_file = os.path.split(img_path)
        img_id, img_ext = os.path.splitext(img_file)

        rel_dir = os.path.relpath(img_dir, input_path)
        out_dir = os.path.join(output_path, rel_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        img = load_image(img_path)
        converter(out_dir, img_id, img)


def numpy_converter(out_dir, img_id, img):
    arr_path = os.path.join(out_dir, f'{img_id}.npy')
    np.save(arr_path, np.array(img, dtype='uint8'))


def jpeg_converter(quality):
    def converter(out_dir, img_id, img):
        jpg_path = os.path.join(out_dir, f'{img_id}.jpg')
        img.save(jpg_path, format='JPEG', quality=quality)
    return converter


def main(args):
    if args.conversion == 'numpy':
        convert(args.indir, args.outdir, numpy_converter, extensions=['png', 'jpg'])
    else:
        convert(args.indir, args.outdir, jpeg_converter(args.jpeg_quality), extensions=['png'])


def parser():
    parser = argparse.ArgumentParser(description='DIV2K image to converter')

    parser.add_argument('conversion', type=str, choices=['numpy', 'jpeg'],
                        help='conversion to perform')
    parser.add_argument('-i', '--indir', type=str, required=True,
                        help='path to DIV2K images')
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help='directory where converted image files are stored')
    parser.add_argument('--jpeg-quality', type=int, default=75,
                        help='JPEG image quality')

    return parser


if __name__ == '__main__':
    main(parser().parse_args())
