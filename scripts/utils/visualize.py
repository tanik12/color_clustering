import numpy as np
import cv2
import sys

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
