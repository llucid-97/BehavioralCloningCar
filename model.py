from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import keras
import tensorflow as tf

"""
Model definition only. Stages:
    1: Pre processing
    2: Convolutional Front End
    3: Dense Back end
"""
input_shape_x = 320
input_shape_y = 160
base_model = applications.MobileNet(include_top=False, input_shape=(input_shape_x, input_shape_y, 3))
print(base_model.summary())
in0 = keras.layers.Input(shape=[160, 320, 3])  # input

# 1: Preprocessing-----------------------------------------
prep = keras.layers.Lambda(
    lambda x: tf.image.rgb_to_hsv(x),
    name="RGB_to_HSV")(in0)

prep = keras.layers.Cropping2D(
    ((50, 20), (0, 0)),
    name="crop")(prep)

prep = keras.layers.BatchNormalization(
    name="BatchNorm_1")(prep)

# 2: Convolutional feature extraction-----------------------------------------

conv = keras.layers.Conv2D(
    32, 3, strides=(2, 2),
    name="Feat_Conv_1",
    activation='relu')(prep)

conv = keras.layers.BatchNormalization()(conv)

conv = keras.layers.SeparableConv2D(
    64, 3, activation='relu',
    padding='same',
    name="Feat_Sep_1")(conv)
conv = keras.layers.BatchNormalization()(conv)

# 3: Dense -----------------------------------------

net = keras.layers.Flatten()(conv)

net = keras.layers.Dropout(0.5)(net)

net = keras.layers.Dense(1, name="Raw_Output")(net)

driveNet = Model(in0, net)

# -----------------------------------------
driveNet.compile(
    optimizer='rmsprop',
    loss='mean_squared_error',
    metrics=['accuracy']
)
