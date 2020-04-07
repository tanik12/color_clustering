import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans

from utils.data_loader import load_image
from utils.hist import centroid_histogram
from utils.visualize import plot_colors

def main(img):
    N=7
    sample_img=[i[int(w/10):int(N*w/10), int(h/10):int(N*h/10)] for i in img]
    sample_img=[i.reshape(-1,3) for i in sample_img]
    color=[]
    for i in sample_img:
        clt = KMeans(n_clusters = 3)
        clt.fit(i)
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

if __name__ == "__main__":
    img = load_image("/Users/gisen/git/color_clustering/data")
    img=np.reshape(img, (len(img),150,150,3)) 
    b,w,h,c=img.shape
    main(img)

#    hstack=np.hstack(img)
