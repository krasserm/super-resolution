import os
import numpy as np

from keras.utils.data_utils import Sequence

DOWNGRADES = ['bicubic', 'bicubic_jpeg_75', 'bicubic_jpeg_90', 'unknown']


class DIV2KSequence(Sequence):
    def __init__(self,
                 path,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 image_ids=None,
                 random_rotate=True,
                 random_flip=True,
                 random_crop=True,
                 crop_size=96,
                 batch_size=16):
        """
        Sequence over a DIV2K subset.

        Reads DIV2K images that have been converted to numpy arrays with convert.py.

        :param path: path to DIV2K dataset with images stored as numpy arrays.
        :param scale: super resolution scale, either 2, 3 or 4.
        :param subset:  either 'train' or 'valid', referring to training and validation subset, respectively.
        :param downgrade: downgrade operator, see DOWNGRADES.
        :param image_ids: list of image ids to use from the specified subset. Default is None which means
                          all image ids from the specified subset.
        :param random_rotate: if True images are randomly rotated by 0, 90, 180 or 270 degrees.
        :param random_flip: if True images are randomly flipped horizontally.
        :param random_crop: if True images are randomly cropped.
        :param crop_size: size of crop window in HR image. Only used if random_crop=True.
        :param batch_size: size of generated batches.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} doesn't exist")
        if scale not in [2, 3, 4]:
            raise ValueError('scale must be 2, 3 or 4')
        if subset not in ['train', 'valid']:
            raise ValueError("subset must be 'train' or 'valid'")
        if downgrade not in DOWNGRADES:
            raise ValueError(f"downgrade must be in {DOWNGRADES}")
        if not random_crop and batch_size != 1:
            raise ValueError('batch_size must be 1 if random_crop=False')

        self.path = path
        self.scale = scale
        self.subset = subset
        self.downgrade = downgrade

        if image_ids is None:
            if subset == 'train':
                self.image_ids = range(1, 801)
            else:
                self.image_ids = range(801, 901)
        else:
            self.image_ids = image_ids

        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.batch_size = batch_size

    def __getitem__(self, index):
        if self.batch_size == 1:
            return self._batch_1(self.image_ids[index])
        else:
            beg = index * self.batch_size
            end = (index + 1) * self.batch_size
            return self._batch_n(self.image_ids[beg:end])

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def _batch_1(self, id):
        lr, hr = self._pair(id)

        return np.expand_dims(np.array(lr, dtype='uint8'), axis=0), \
               np.expand_dims(np.array(hr, dtype='uint8'), axis=0)

    def _batch_n(self, ids):
        lr_crop_size = self.crop_size // self.scale
        hr_crop_size = self.crop_size

        lr_batch = np.zeros((len(ids), lr_crop_size, lr_crop_size, 3), dtype='uint8')
        hr_batch = np.zeros((len(ids), hr_crop_size, hr_crop_size, 3), dtype='uint8')

        for i, id in enumerate(ids):
            lr, hr = self._pair(id)
            lr_batch[i] = lr
            hr_batch[i] = hr

        return lr_batch, hr_batch

    def _pair(self, id):
        lr_path = self._lr_image_path(id)
        hr_path = self._hr_image_path(id)

        lr = np.load(lr_path)
        hr = np.load(hr_path)

        if self.random_crop:
            lr, hr = _random_crop(lr, hr, self.crop_size, self.scale)
        if self.random_flip:
            lr, hr = _random_flip(lr, hr)
        if self.random_rotate:
            lr, hr = _random_rotate(lr, hr)

        return lr, hr

    def _hr_image_path(self, id):
        return os.path.join(self.path, f'DIV2K_{self.subset}_HR', f'{id:04}.npy')

    def _lr_image_path(self, id):
        return os.path.join(self.path, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}', f'{id:04}x{self.scale}.npy')


def cropped_sequence(path, scale, subset, downgrade, image_ids=None, batch_size=16):
    return DIV2KSequence(path=path, scale=scale, subset=subset, downgrade=downgrade, image_ids=image_ids,
                         batch_size=batch_size, crop_size=48 * scale)


def fullsize_sequence(path, scale, subset, downgrade, image_ids=None):
    return DIV2KSequence(path=path, scale=scale, subset=subset, downgrade=downgrade, image_ids=image_ids,
                         batch_size=1, random_rotate=False, random_flip=False, random_crop=False)


def _random_crop(lr_img, hr_img, hr_crop_size, scale):
    lr_crop_size = hr_crop_size // scale

    lr_w = np.random.randint(lr_img.shape[1] - lr_crop_size + 1)
    lr_h = np.random.randint(lr_img.shape[0] - lr_crop_size + 1)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def _random_flip(lr_img, hr_img):
    if np.random.rand() > 0.5:
        return np.fliplr(lr_img), np.fliplr(hr_img)
    else:
        return lr_img, hr_img


def _random_rotate(lr_img, hr_img):
    k = np.random.choice(range(4))
    return np.rot90(lr_img, k), np.rot90(hr_img, k)
