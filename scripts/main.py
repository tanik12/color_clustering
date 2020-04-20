import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from mpl_toolkits.mplot3d import Axes3D

from utils.data_loader import load_image, color_list
from utils.hist import centroid_histogram
from utils.visualize import plot_colors, plot, plot_lab_3candidates, plot_lab_final_candidates
from utils.similarity import cos_sim, similarity_calculate, similarity_calculate_norm
from utils.evaluation import accuracy

def main(img, label_arrays):
    label_dict = {"red" : 0, "blue" : 1, "yellow" : 2, "unknown" : 3}
    cmp_lab_arr = color_list()

    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("L")

    sample_img = [i.reshape(-1,3) for i in img]
    color=[]
    
    label_arrays_tmp = np.array([])
    res_array = np.array([])
    for num, (i, label) in enumerate(zip(sample_img, label_arrays)):
        #################
        #### Lab計算 #####
        clt = KMeans(n_clusters = 3)
        clt.fit(i)
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)
        
        colors = (clt.cluster_centers_).astype("uint8")
        prob = hist.reshape(-1, 1)
        print("RGB座標：", colors, colors.shape)
        print("割合   ：", prob)
                
        res = colors[np.newaxis, :,:]
        bar_lab = rgb2lab(res) # LAB値に変換

        print("Lab座標：", bar_lab[0], bar_lab.shape, bar_lab[0].shape)
        idx = np.where( (np.abs(bar_lab[0][:, 1]) > 10) | (np.abs(bar_lab[0][:, 2]) > 10) )
        lab_candidate = bar_lab[0][idx]

        if lab_candidate.shape[0] == 0:
            continue

        #plot_lab_3candidates(bar)   #debug用 --> クラスタリング結果の3つの候補を出す。
        #plot_lab_final_candidates(lab_candidate) #debug用 --> クラスタリング結果の3つの候補のうち最終的な候補を出す。上記のidxにより抽出。
        label_arrays_tmp = np.append(label_arrays_tmp, num)
        #################

        ######################
        #### 類似度計算計算 ####
        #pre_label, lab_info  = similarity_calculate_norm(cmp_lab_arr, lab_candidate)
        pre_label, lab_info = similarity_calculate(cmp_lab_arr, lab_candidate)
        #####################

        plot(lab_info, label, fig, ax)
        print("----------", pre_label)
        res_array = np.append(res_array, label_dict[pre_label])

    ###plt.show()

    accuracy(label_arrays[label_arrays_tmp.astype(np.int)], res_array)
    #accuracy(label_arrays, res_array)

if __name__ == "__main__":
    #img, label_arrays = load_image("/Users/gisen/git/color_clustering/data")
    img, label_arrays = load_image("/Users/gisen/git/color_clustering/data_evaluation")
    print("教師ラベルの総数: ", label_arrays.shape)
    img = np.reshape(img, (len(img),150,150,3))
    b,w,h,c = img.shape
    main(img, label_arrays)
