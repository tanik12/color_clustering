import numpy as np
import sys

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def inner_product(v1, v2):
    return np.dot(v1, v2)

def norm_vec(v1, v2):
    test = np.ones((4,3)) * v2
    #v3 = np.linalg.norm(test-v1)
    #v3 = test-v1

    tmp = np.array([])
    for i in range(4):
        #v3 = np.linalg.norm(test[i]-v1[i])
        v3 = np.linalg.norm(test[i]-v1[i])
        tmp = np.append(tmp, v3)
    
    return tmp.reshape(-1, 1)

def similarity_calculate(comparison_arrays, lab_arrays):
    for lab_array in lab_arrays:
        lab_array = np.array([lab_array])

        tmp = cos_sim(comparison_arrays, lab_array.reshape(3, 1))
        max_val = np.max(tmp, axis=0)
        max_index = np.argmin(tmp, axis=0)

        if max_index == 0 and max_val >= 0.45:
            class_name = "red"
        elif (max_index == 1 or max_index == 2) and max_val >= 0.45:
            class_name = "blue"
        elif max_index == 3 and max_val >= 0.45:
            class_name = "yellow"
        else:
            class_name = "unknown"

        print("推論結果: ", max_val, class_name)
    print("=======\n")

def similarity_calculate_norm(comparison_arrays, lab_arrays):
    result_diff_val = 1000 #1000はdummy data
    result_label = ""
    for lab_array in lab_arrays:
        lab_array = np.array([lab_array])

        tmp = norm_vec(comparison_arrays, lab_array)
        min_val = np.min(tmp, axis=0)
        min_index = np.argmin(tmp, axis=0)

        if min_index == 0:
            class_name = "red"
        elif min_index == 1 or min_index == 2:
            class_name = "blue"
        elif min_index == 3:
            class_name = "yellow"
        else:
            class_name = "unknown"

        print("推論結果: ", min_val, class_name)
        if result_diff_val > min_val[0]:
            result_diff_val = min_val[0]
            result_label = class_name
        
    print("最終的な色識別の判定結果：", result_label)
    print("=======\n")
