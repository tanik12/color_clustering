import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def norm2d(x,y,sigma):
    Z = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    #Z = np.where(Z > 0.003, 1000, Z)

    return Z

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # histのパーセンテージ分まで色付け。colorの三次元情報が色情報。
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    #opencvによって色付けした結果をreturn
    return bar

#RGB情報の可視化(3D)max_index >= 3 and max_index < 4)
def plot(lab, label, fig, ax):
    print(lab.reshape(1,-1), int(label))
    lab = lab.reshape(1,-1)
    label = int(label)
    aaa = norm2d(lab[0][1], lab[0][2], 0.5)

    if label == 0:
        ax.scatter(lab[0][1], lab[0][2], lab[0][0], s = 10, c = "red")
    elif label==1:
        ax.scatter(lab[0][1], lab[0][2], lab[0][0], s = 10, c = "blue")
    elif label == 2:
        ax.scatter(lab[0][1], lab[0][2], lab[0][0], s = 10, c = "yellow")
    else:
        ax.scatter(lab[0][1], lab[0][2], lab[0][0], s = 10, c = "black")