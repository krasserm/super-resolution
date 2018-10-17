import os
import tempfile


def load_model(path):
    import train
    return train._load_model(path)


def copy_weights(from_model, to_model, by_name=True):
    with tempfile.TemporaryDirectory(dir='.') as d:
        path = os.path.join(d, 'weights.h5')
        from_model.save_weights(path)
        to_model.load_weights(path, by_name=by_name)
