import tensorflow as tf

from contextlib import contextmanager
from PIL import Image

from keras import backend as K
from keras.utils.data_utils import OrderedEnqueuer


@contextmanager
def concurrent_generator(sequence, num_workers=8, max_queue_size=32, use_multiprocessing=False):
    enqueuer = OrderedEnqueuer(sequence, use_multiprocessing=use_multiprocessing)
    try:
        enqueuer.start(workers=num_workers, max_queue_size=max_queue_size)
        yield enqueuer.get()
    finally:
        enqueuer.stop()


def init_session(gpu_memory_fraction):
    K.tensorflow_backend.set_session(tensorflow_session(gpu_memory_fraction=gpu_memory_fraction))


def reset_session(gpu_memory_fraction):
    K.clear_session()
    init_session(gpu_memory_fraction)


def tensorflow_session(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    return tf.Session(config=config)


def load_image(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
