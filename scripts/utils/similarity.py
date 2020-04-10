import numpy as np

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
