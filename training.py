from model import driveNet
from loadData import loadData, prepareData, getValidationSet
import os
from debugModel import tensorBoard
import keras

"""
training data must be in directories named data*/
validation data must be in directory named validation

Desired checkpoint weights to restore must be named ./checkpoints/dummy.h5
Re-Trains from scratch if not found
"""

X, y = prepareData(loadData())
X_val,y_val = getValidationSet()

checkpoints = keras.callbacks.ModelCheckpoint(
    "./checkpoints/weights.{epoch:02d}.hdf5",
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='auto',
    period=1)


if os.path.exists("./checkpoints/dummy.h5"):
    driveNet.load_weights("./checkpoints/dummy.h5", True, True)

driveNet.fit(x=X, y=y, batch_size=100, epochs=3,
             validation_data=(X_val,y_val),
             shuffle=True,
             callbacks=[tensorBoard, checkpoints])

driveNet.save_weights('./checkpoints/final_weights.h5')
driveNet.save('model.h5')
