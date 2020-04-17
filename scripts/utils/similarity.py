import numpy as np
import sys

#Labの中で、類似度計算をする際にはLは使わない方が良いかもしれない。
#a,bで値がどれか10以下であった場合は、unknownにすること。
#クラスタリングした際に三つの色候補が出るがa,bで値がどれか10以下が出た場合、類似度計算に使用しないベクトルとして認識するすること.

#cos類似度
def cos_sim(v1, v2):
    ####aaa = np.ones((3,1))
    ####aaa[0, 0] = aaa[0, 0]/100
    ####aaa[0, 1] = aaa[0, 1]/128
    ####aaa[0, 2] = aaa[0, 2]/128
    ###print("000") 
    ###print(v1)
    ###print(v2)
    ###aaa = np.array([100, 128, 128])
    ###v2 = v2 / aaa.reshape(-1,1)
    ###v1 = v1 / aaa
    ###print(v1)
    ###print("111") 
    ###print(v2)
    ###print("222")
    #return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.dot(v1[:, 1:], v2[1:, :]) / (np.linalg.norm(v1[:, 1:]) * np.linalg.norm(v2[1:, :]))

#内積
def inner_product(v1, v2):
    return np.dot(v1, v2)

#ユークリッド距離
def norm_vec(v1, v2):
    v2_rep = np.ones((4,3)) * v2
    #v2_rep = np.ones((6,3)) * v2

    tmp = np.array([])
    #for i in range(6):
    for i in range(4):
        #v3 = np.linalg.norm(v2_rep[i]-v1[i])
        v3 = np.linalg.norm(v2_rep[i][1:]-v1[i][1:])
        print("-------------------->>> ", v3)
        tmp = np.append(tmp, v3)
    
    return tmp.reshape(-1, 1)

#cos類似度指標による色検出
def similarity_calculate(comparison_arrays, lab_arrays):
    result_diff_val = 0 #0はdummy data
    result_label = ""
    tmptmp = np.zeros((3,1))
    for lab_array in lab_arrays:
        lab_array = np.array([lab_array])

        tmp = cos_sim(comparison_arrays, lab_array.reshape(3, 1))
        max_val = np.max(tmp, axis=0)
        max_index = np.argmax(tmp, axis=0)

        #if max_index < 2 and max_val >= 0.4:
        if max_index < 1 and max_val >= 0.2:
            class_name = "red"
        #elif (max_index >= 2 and max_index < 4) and max_val >= 0.4:
        elif (max_index >= 1 and max_index < 3) and max_val >= 0.2:
            class_name = "blue"
        #elif (max_index >= 4 and max_index < 5) and max_val >= 0.4:
        elif (max_index >= 3 and max_index < 4) and max_val >= 0.2:
            class_name = "yellow"
        else:
            class_name = "unknown"
        
        if result_diff_val < max_val[0]:
            result_diff_val = max_val[0]
            result_label = class_name
            tmptmp = lab_array.reshape(3, 1)
            if max_val < 0.2:
                result_label = "unknown"
        
        if result_label == "":
            result_label = "unknown"

###        if max_index == 0 and max_val >= 0.45:
###            class_name = "red"
###        elif (max_index == 1 or max_index == 2) and max_val >= 0.45:
###            class_name = "blue"
###        elif max_index == 3 and max_val >= 0.45:
###            class_name = "yellow"
###        else:
###            class_name = "unknown"

        print("推論結果: ", max_val, class_name)
    print("=======\n")
    print("最終的な色識別の判定結果：", result_label)
    print("=======\n")
    return result_label, tmptmp

#ユークリッド距離指標による色検出
def similarity_calculate_norm(comparison_arrays, lab_arrays):
    result_diff_val = 1000 #1000はdummy data
    result_label = ""
    for lab_array in lab_arrays:
        lab_array = np.array([lab_array])

        tmp = norm_vec(comparison_arrays, lab_array)
        min_val = np.min(tmp, axis=0)
        min_index = np.argmin(tmp, axis=0)

        if min_index < 1:
            class_name = "red"
        elif 1 <= min_index < 3:
            class_name = "blue"
        elif 3 <= min_index < 5:
            class_name = "yellow"
        else:
            class_name = "unknown"

        if result_diff_val > min_val[0]:
            result_diff_val = min_val[0]
            result_label = class_name
            tmptmp = lab_array.reshape(3, 1)
            if result_diff_val > 80:
                result_label = "unknown"

        print("推論結果: ", min_val, class_name)

    print("最終的な色識別の判定結果：", result_label)
    print("=======\n")
    return result_label, tmptmp
