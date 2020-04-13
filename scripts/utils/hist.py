import numpy as np

def centroid_histogram(clt):
    # クラスターの数からbinの決定
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # 割合
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist
