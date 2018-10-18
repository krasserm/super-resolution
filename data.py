import os
import numpy as np

from PIL import Image


class DIV2KDataset:
    def __init__(self, path, subset='train', image_ids=None, cache_images=True):
        """
        Represents a subset of the DIV2K dataset.

        :param path: path that contains the extracted DIV2K archives from https://data.vision.ee.ethz.ch/cvl/DIV2K/
        :param subset: either 'train' or 'valid', referring to training and validation subset, respectively.
        :param image_ids: list of image ids to use from the specified subset. Default is None which means
               all image ids from the specified subset.
        :param cache_images: whether to cache loaded images in memory. Default is True.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} doesn't exist")
        if subset not in ['train', 'valid']:
            raise ValueError("subset must be 'train' or 'valid'")

        self.path = path
        self.subset = subset
        self.cache_images = cache_images
        self.cache = {}

        if image_ids is None:
            if subset == 'train':
                self.image_ids = range(1, 801)
            else:
                self.image_ids = range(801, 901)
        else:
            self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def pair_generator(self, downgrade='bicubic', scale=2, repeat=True):
        """
        Returns a generator that yields (LR, HR) PIL image pairs.

        :param downgrade: downgrade operator, either 'bicubic' or 'unknown'.
        :param scale: super resolution scale, either 2, 3 or 4.
        :param repeat: True if generator shall repeatedly loop over this dataset.
        :return: (LR, HR) PIL image pair generator.
        """

        if downgrade not in ['bicubic', 'unknown']:
            raise ValueError("downgrade must be 'bicubic' or 'unknown'")
        if scale not in [2, 3, 4]:
            raise ValueError('scale must be 2, 3 or 4')

        while True:
            for id in self.image_ids:
                hr_path = self._hr_image_path(id)
                lr_path = self._lr_image_path(downgrade, scale, id)

                hr_img = self._image(hr_path)
                lr_img = self._image(lr_path)

                yield lr_img, hr_img

            if not repeat:
                break

    def _hr_image_path(self, id):
        return os.path.join(self.path, f'DIV2K_{self.subset}_HR', f'{id:04}.png')

    def _lr_image_path(self, downgrade, scale, id):
        return os.path.join(self.path, f'DIV2K_{self.subset}_LR_{downgrade}', f'X{scale}', f'{id:04}x{scale}.png')

    def _image(self, path):
        img = self.cache.get(path)
        if not img:
            img = load_image(path)
            if self.cache_images:
                self.cache[path] = img
        return img


def load_image(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def random_transform(generator, random_rotate=True, random_flip=True, random_crop=True, crop_size=96):
    """
    Decorator for random transformations of image pairs yielded from generator.

    :param generator: generator that yields (LR, HR) PIL image pairs.
    :param random_rotate: True if random 0, 90, 180, 270 degree rotations shall be generated.
    :param random_flip: True if random horizontal flips shall be generated.
    :param random_crop: True if random crops shall be generated.
    :param crop_size: size of crop window in HR image. Only used if random_crop=True.
    :return: generator that yields transformed (LR, HR) image pairs.
    """

    for lr, hr in generator:
        if random_crop:
            lr, hr = _random_crop(lr, hr, crop_size, scale=hr.width // lr.width)
        if random_flip:
            lr, hr = _random_flip(lr, hr)
        if random_rotate:
            lr, hr = _random_rotate(lr, hr)
        yield lr, hr


def batch(generator, batch_size):
    """
    Decorator for batching PIL image pairs yielded form generator. PIL images are converted to numpy arrays.

    :param generator: generator that yields (LR, HR) PIL image pairs.
    :param batch_size: maximum size of batches to be generated.
    :return: generator that yields (LR, HR) image batch pairs.
    """

    lr_batch = []
    hr_batch = []

    for lr, hr in generator:
        lr_batch.append(np.expand_dims(lr, axis=0))
        hr_batch.append(np.expand_dims(hr, axis=0))

        if len(lr_batch) == batch_size:
            yield np.concatenate(lr_batch), np.concatenate(hr_batch)
            lr_batch = []
            hr_batch = []

    if lr_batch:
        yield np.concatenate(lr_batch), np.concatenate(hr_batch)


def random_generator(path, subset, downgrade, scale, batch_size=16, image_ids=None, cache_images=True):
    """
    Convenience generator for randomly cropped image pairs.
    """

    ds = DIV2KDataset(path=path, subset=subset, image_ids=image_ids, cache_images=cache_images)
    gen = ds.pair_generator(downgrade=downgrade, scale=scale)
    gen = random_transform(gen, crop_size=48 * scale)
    gen = batch(gen, batch_size=batch_size)
    return gen


def fullsize_generator(path, subset, downgrade, scale, image_ids=None, cache_images=True):
    """
    Convenience generator for full-size image pairs and batch size 1.
    """

    ds = DIV2KDataset(path=path, subset=subset, image_ids=image_ids, cache_images=cache_images)
    gen = ds.pair_generator(downgrade=downgrade, scale=scale)
    gen = batch(gen, batch_size=1)
    return gen


def _random_crop(lr_img, hr_img, hr_crop_size, scale):
    lr_crop_size = hr_crop_size // scale

    lr_w = np.random.randint(lr_img.width - lr_crop_size + 1)
    lr_h = np.random.randint(lr_img.height - lr_crop_size + 1)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img.crop((lr_w, lr_h, lr_w + lr_crop_size, lr_h + lr_crop_size))
    hr_img_cropped = hr_img.crop((hr_w, hr_h, hr_w + hr_crop_size, hr_h + hr_crop_size))

    return lr_img_cropped, hr_img_cropped


def _random_flip(lr_img, hr_img):
    if np.random.rand() > 0.5:
        return lr_img.transpose(Image.FLIP_LEFT_RIGHT), hr_img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return lr_img, hr_img


def _random_rotate(lr_img, hr_img):
    rot = np.random.choice((None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270))
    if not rot:
        return lr_img, hr_img
    else:
        return lr_img.transpose(rot), hr_img.transpose(rot)
