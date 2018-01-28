import tensorflow as tf
import keras
from model import driveNet
import keras.backend as K

tensorBoard = keras.callbacks.TensorBoard(
    log_dir='/tmp/driveNet',
    histogram_freq=0,
    batch_size=32,
    write_graph=True,
    write_grads=False,
    write_images=False,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None)

driveNet.summary()