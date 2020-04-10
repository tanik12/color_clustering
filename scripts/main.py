import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

from utils.data_loader import load_image
from utils.hist import centroid_histogram
from utils.visualize import plot_colors

def color_list():
    red_lab = np.array([56, 77, 32])
    aqua_lab = np.array([91, -48, -10])    
    green_lab = np.array([73, -61, 30])
    yellow_lab = np.array([86, -7, 86])
    
    colors_lab = np.vstack((red_lab, aqua_lab, green_lab, yellow_lab))
    return colors_lab

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def similarity_calculate(comparison_arrays, lab_arrays):
    for lab_array in lab_arrays:
        lab_array = np.array([lab_array])
        tmp = cos_sim(comparison_arrays, lab_array.reshape(3, 1))
        max_val = np.max(tmp, axis=0)
        max_index = np.argmax(tmp, axis=0)
        if max_index == 0 and max_val >= 0.45:
            class_name = "red"
        elif (max_index == 1 or max_index == 2) and max_val >= 0.45:
            class_name = "blue"
        elif max_index == 3 and max_val >= 0.45:
            class_name = "yellow"
        else:
            class_name = "unknown"

        print("推論結果: ", max_val, class_name)
    print("=======")

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
