from model import driveNet

"""
Load a checkpoint and save as a model meta-graph
"""


driveNet.load_weights("./checkpoints/weights.01.hdf5", True, True)
driveNet.save('model.h5')