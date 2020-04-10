import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_image(image_file):
    # cv2 load images as BGR
    image_bgr=[cv2.imread(image_file+'/'+i) for i in os.listdir(image_file)]
    image_rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in image_bgr]
    image_rgb = [cv2.resize(i, (150, 150)) for i in image_rgb]
    return image_rgb

def color_list():
    red_lab = np.array([56, 77, 32])
    aqua_lab = np.array([91, -48, -10])    
    green_lab = np.array([73, -61, 30])
    yellow_lab = np.array([86, -7, 86])
    
    colors_lab = np.vstack((red_lab, aqua_lab, green_lab, yellow_lab))
    return colors_lab

if __name__ == "__main__":
    img = load_image("/Users/gisen/git/color_clustering/data")
    print(type(img))
    img=np.reshape(img, (len(img),150,150,3))
    print(img.shape)
    hstack=np.hstack(img)
    print(hstack.shape)
    plt.imshow(hstack)
    plt.show()
