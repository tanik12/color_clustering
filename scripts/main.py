import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

from utils.data_loader import load_image, color_list
from utils.hist import centroid_histogram
from utils.visualize import plot_colors
from utils.similarity import cos_sim, similarity_calculate

def main(img):
    cmp_lab_arr = color_list()
    sample_img = [i.reshape(-1,3) for i in img]
    color=[]
    
    for i in sample_img:
        #################
        #### Lab計算 #####
        clt = KMeans(n_clusters = 3)
        clt.fit(i)
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)
        
        colors = (clt.cluster_centers_).astype("uint8")
        prob = hist.reshape(-1, 1)
        print("RGB座標：", colors)
        print("確率   ：", prob)
        
        plt.figure()
        plt.axis("off")
        plt.imshow(bar) #debug用
        plt.show()      #debug用
        
        res = colors[np.newaxis, :,:]
        bar_lab = rgb2lab(res) # LAB値に変換
        print("Lab座標：", bar_lab[0])
        print("比較対象")
        print(cmp_lab_arr)
        print("----------")
        #################

        ######################
        #### 類似度計算計算 ####
        similarity_calculate(cmp_lab_arr, bar_lab[0])
        #####################

if __name__ == "__main__":
    img = load_image("/Users/gisen/git/color_clustering/data")
    img=np.reshape(img, (len(img),150,150,3)) 
    b,w,h,c=img.shape
    main(img)
