# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 04:55:21 2021

@author: Himanshu
"""

import pydicom as dicom
import matplotlib.pyplot as plt
import os
import cv2
import PIL # optional
import pandas as pd
import csv
# make it True if you want in PNG format
PNG = False
# Specify the .dcm folder path
folder_path = "C:/Users/Himanshu/Desktop/data/Images/"
# Specify the .jpg/.png folder path
jpg_folder_path = "C:/Users/Himanshu/Desktop/data/JPG/"
"""
images_path = os.listdir(folder_path)
for n, image in enumerate(images_path):
    ds = dicom.dcmread(os.path.join(folder_path, image))
    print("Pixel Data" in ds)
    pixel_array_numpy = ds.pixel_array
#    if PNG == False:
#       image = image.replace('.dcm', '.jpg')
#    else:
#        image = image.replace('.dcm', '.png')
    cv2.imwrite(os.path.join(jpg_folder_path, image + '.jpg'), pixel_array_numpy)
    if n % 50 == 0:
        print('{} image converted'.format(n))
        
"""

# specify your image path
image_path = 'C:/Users/Himanshu/Desktop/data/Images/00962781'
ds = dicom.dcmread(image_path)
plt.imshow( ds.pixel_array)

plt.show()