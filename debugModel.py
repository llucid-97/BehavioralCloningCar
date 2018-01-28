import tensorflow as tf
import keras
from driveNet import driveNet
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

if __name__ == "__main__":
    from keras import applications
    import numpy as np

    in0 = keras.layers.Input(shape=[160, 320, 3])

    # Preprocessing
    prep = keras.layers.Lambda(
        lambda x: tf.image.rgb_to_hsv(x)
        , name="RGB_to_HSV")(in0)
    prep = keras.layers.Cropping2D(
        ((50, 20), (0, 0)), name="crop")(prep)
    prep = keras.layers.BatchNormalization(name="BatchNorm_1")(prep)

    model = applications.MobileNet(input_shape=[160, 160, 3], include_top=False,
                                   weights='imagenet')

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    model.summary()

    layer_name = 'conv_dw_11'
    filter_index = 0

    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, input_img)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.random.random((1,3,img_width, img_height)) *20 +128