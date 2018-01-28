import numpy as np
import csv
import cv2
import os
import matplotlib.pyplot as plt


def loadData(dataDir=None):
    """

    :param dataDir: String path to datset
    :return:
    """
    training_data = {
        "cam_center": [],
        "cam_left": [],
        "cam_right": [],
        "steering_angle": [],

    }
    if dataDir is None:
        tmp = [f for f in os.listdir(os.getcwd()) if os.path.isdir(f)]
        dataDirLst = [os.path.join(os.getcwd(), k) for k in tmp if 'data' in k]

    else:
        dataDirLst = [dataDir]

    for dataDir in dataDirLst:
        with open(os.path.join(dataDir, 'driving_log.csv')) as f:
            reader = csv.reader(f)
            lines = list(reader)
            for i, line in enumerate(lines):
                training_data["cam_center"].append(cv2.cvtColor(cv2.imread(line[0]), cv2.COLOR_BGR2RGB))
                training_data["cam_left"].append(cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2RGB))
                training_data["cam_right"].append(cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2RGB))
                training_data["steering_angle"].append(line[3])

        return training_data


def prepareData(dataDict):
    angle_offset = 0.1

    X_center = np.array(dataDict["cam_center"])
    y_center = np.array(dataDict["steering_angle"], dtype=np.float32)

    X_left = np.array(dataDict["cam_left"])
    y_left = y_center + angle_offset

    X_right = np.array(dataDict["cam_right"])
    y_right = y_center - angle_offset

    # crop
    X_data = np.vstack((X_center, X_left, X_right))
    y_data = np.concatenate((y_center, y_left, y_right))

    # flip
    X_flip = []
    for i in range(X_data.shape[0]):
        X_flip.append(cv2.flip(X_data[i], 1))

    X_flip = np.array(X_flip)
    y_flip = np.negative(y_data)
    print(X_flip.shape)
    print(y_flip.shape)

    X_data = np.vstack((X_data, X_flip))
    y_data = np.concatenate((y_data, y_flip))
    print(X_data.shape[0])
    print(y_data.shape)

    return X_data, y_data

def getValidationSet():
    validationSet = loadData('./validation')
    X_val= np.array(validationSet["cam_center"])
    y_val = np.array(validationSet["steering_angle"], dtype=np.float32)

    return X_val,y_val
