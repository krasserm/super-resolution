import os

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard


class ModelCheckpointAfter(ModelCheckpoint):
    def __init__(self, epoch, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
        self.after_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.after_epoch:
            super().on_epoch_end(epoch, logs)


def model_checkpoint_after(epoch, path, monitor, save_best_only):
    pattern = os.path.join(path, 'epoch-{epoch:03d}-psnr-{' + monitor + ':.4f}.h5')
    return ModelCheckpointAfter(epoch, filepath=pattern, monitor=monitor,
                                save_best_only=save_best_only, mode='max')


def learning_rate(step_size, decay, verbose=1):
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr

    return LearningRateScheduler(schedule, verbose=verbose)


def tensor_board(path):
    return TensorBoard(log_dir=os.path.join(path, 'log'), write_graph=False)
