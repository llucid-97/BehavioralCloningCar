from driveNet import driveNet
from loadData import loadData, prepareData, getValidationSet
import os
from debugModel import tensorBoard
import keras

# driveNet.load_weights("./checkpoints/weights.best.hdf5", True, True)
# driveNet.save('model.h5')
# exit(0)
X, y = prepareData(loadData())
X_val,y_val = getValidationSet()

checkpoints = keras.callbacks.ModelCheckpoint(
    "./checkpoints/weights.{epoch:02d}.hdf5",
    monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', period=1)


if os.path.exists("./checkpoints/dummy.h5"):
    driveNet.load_weights("./checkpoints/dummy.h5", True, True)

driveNet.fit(x=X, y=y, batch_size=100, epochs=3,
             validation_data=(X_val,y_val),
             shuffle=True,
             callbacks=[tensorBoard, checkpoints])
driveNet.save_weights('./checkpoints/final_weights.h5')
driveNet.save('model.h5')
