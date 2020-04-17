import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
from mpl_toolkits.mplot3d import Axes3D

from utils.data_loader import load_image, color_list
from utils.hist import centroid_histogram
from utils.visualize import plot_colors, plot
from utils.similarity import cos_sim, similarity_calculate, similarity_calculate_norm

label_dict = {"red" : 0, "blue" : 1, "yellow" : 2, "unknown" : 3}
label_list_ = [0, 0, 0, 0] 

def accuracy(tl_label, pre_res):
    count = 0
    count11 = 0
    count2222 = 0
    for tl, pre in zip(tl_label, pre_res):
        if int(tl) == int(pre):
            count += 1
            if int(tl) == int(pre) and int(pre) != 3:
                count11 += 1
        else:
            if tl == 0:
                label_list_[0] += 1
            elif tl == 1:
                label_list_[1] += 1
            elif tl == 2:
                label_list_[2] += 1
            elif tl == 3:
                label_list_[3] += 1

        if tl != 3:
            count2222 += 1

    print("正解率：", str(count) + "/" + str(len(tl_label)))
    print("精度は", count/len(tl_label) * 100, "%です。")
    print("総数：", len(tl_label))
    print("間違い label_list_ : ", label_list_)
    print("あわわわ label_list_ : ", count11/count2222 )

def main(img, label_arrays):
    cmp_lab_arr = color_list()

    ###print(img.shape)
    ###img_reshape = img.reshape(img.shape[0], -1, img.shape[3])
    ###print(img_reshape.shape)
    ###sys.exit()

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
        print("RGB座標：", colors)
        print("割合   ：", prob)
        
        ###plt.figure()
        ###plt.axis("off")
        ###plt.imshow(bar) #debug用
        ###plt.show()      #debug用
        
        res = colors[np.newaxis, :,:]
        bar_lab = rgb2lab(res) # LAB値に変換

        print("&&&&&&&&&&&&&&&&&&")
        print("Lab座標：", bar_lab[0])
        idx = np.where( (np.abs(bar_lab[0][:, 1]) > 10) | (np.abs(bar_lab[0][:, 2]) > 10) )
        aaatmp = bar_lab[0][idx]
        ###print(aaatmp.shape, aaatmp)
        ###plt.figure()
        ###plt.axis("off")
        ###plt.imshow(aaatmp) #debug用
        ###plt.show()      #debug用

        #print(aaatmp.shape, aaatmp)
        if aaatmp.shape[0] == 0:
            continue
        label_arrays_tmp = np.append(label_arrays_tmp, num)
        print("&&&&&&&&&&&&&&&&&&")
        print(bar_lab[0].shape, aaatmp.shape)
        print("比較対象")
        print(cmp_lab_arr)
        #################

        ######################
        #### 類似度計算計算 ####
        #pre_label, tmptmp  = similarity_calculate_norm(cmp_lab_arr, bar_lab[0])
        pre_label, tmptmp = similarity_calculate(cmp_lab_arr, aaatmp)
        
        #####################
        print('$$$$$$$$')
        plot(tmptmp, label, fig, ax)
        print("----------", pre_label)

        res_array = np.append(res_array, label_dict[pre_label])

    plt.show()
    #print(label_arrays.shape, label_arrays_tmp)
    #print(label_arrays[label_arrays_tmp.astype(np.int)].shape, res_array.shape)

    accuracy(label_arrays[label_arrays_tmp.astype(np.int)], res_array)
    #accuracy(label_arrays, res_array)

if __name__ == "__main__":
    #img, label_arrays = load_image("/Users/gisen/git/color_clustering/data")
    img, label_arrays = load_image("/Users/gisen/git/color_clustering/data_evaluation")
    print("aaa; ", label_arrays.shape)
    img = np.reshape(img, (len(img),150,150,3))
    b,w,h,c = img.shape
    main(img, label_arrays)
