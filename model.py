from keras.models import Model
import keras
import tensorflow as tf

in0 = keras.layers.Input(shape=[160, 320, 3])

# Preprocessing
prep = keras.layers.Lambda(
    lambda x: tf.image.rgb_to_hsv(x)
    , name="RGB_to_HSV")(in0)
prep = keras.layers.Cropping2D(
    ((50, 20), (0, 0)), name="crop")(prep)
prep = keras.layers.BatchNormalization(name="BatchNorm_1")(prep)

# Convolutional feature extraction

conv = keras.layers.Conv2D(32, 3, strides=(2, 2),
                           name="Feat_Conv_1",activation='relu')(prep)
conv = keras.layers.BatchNormalization()(conv)
conv = keras.layers.SeparableConv2D(64, 3, activation='relu',
                                    padding='same',
                                    name="Feat_Sep_1")(conv)
conv = keras.layers.BatchNormalization()(conv)
# conv = keras.layers.SeparableConv2D(128,3,strides=(2,2),
#                                     activation='relu',
#                                     padding='same',
#                                     name="Feat_Sep_2",)(conv)
# conv = keras.layers.SeparableConv2D(128, 3, activation='relu',padding='same',
#                                     name="Feat_Sep_3")(conv)
# conv = keras.layers.SeparableConv2D(256,3,strides=(2,2),padding='same',
#                                     activation='relu',
#                                     name="Feat_Sep_4")(conv)
# conv = keras.layers.SeparableConv2D(256, 3, activation='relu',padding='same',
#                                     name="Feat_Sep_5")(conv)
# conv = keras.layers.SeparableConv2D(512,3,strides=(2,2),padding='same',
#                                     activation='relu',
#                                     name="Feat_Sep_6")(conv)
# for i in range(5):
#     conv = keras.layers.SeparableConv2D(512, 3,strides=(2,2),
#                                         activation='relu',padding='same',
#                                         name="Feat_Sep_"+str(7+i))(conv)
#     conv = keras.layers.Dropout(0.9)(conv)
#
# conv = keras.layers.SeparableConv2D(1024,3,strides=(2,2),padding='same',
#                                     activation='relu',
#                                     name="Feat_Sep_12")(conv)
# conv = keras.layers.SeparableConv2D(1024,3,padding='valid',
#                                     activation='relu',
#                                     name="Feat_Sep_13")(conv)
# Dense
net = keras.layers.Flatten()(conv)
# net = keras.layers.Dense(1000,name="bottleneck",activation='relu')(net)
# net = keras.layers.Dense(1000,name="Deeper")(net)
net = keras.layers.Dropout(0.5)(net)
net = keras.layers.Dense(1,name="Raw_Output")(net)
driveNet = Model(in0, net)

import keras.backend as K


def stepRight(y_true, y_pred):
    # return K.mean(
    #     (((y_true * y_pred) > 0)or(y_true==y_pred))
    # )
    return 0


driveNet.compile(
    optimizer='rmsprop',
    loss='mean_squared_error',
    metrics=['accuracy']
)
