import keras
def MobileNet_Separable():
    conv = keras.layers.SeparableConv2D(64, 3, activation='relu',
                                        padding='same',
                                        name="Feat_Sep_1")(conv)