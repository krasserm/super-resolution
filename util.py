import tensorflow as tf

from tensorflow.keras.backend import set_session
from PIL import Image


def init_session(gpu_memory_fraction):
    set_session(tensorflow_session(gpu_memory_fraction=gpu_memory_fraction))


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
