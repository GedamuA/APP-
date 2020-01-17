# -*- coding: utf-8 -*-
"""Description :
"""
# Gedamu A.  2019/11/13
# Center for future media lab 
from __future__ import print_function, division, absolute_import, unicode_literals
import pdb
import glob
import os
from PIL import Image
import cv2
import numpy as np
from ssim import SSIM
# install ssim packages
# pip install pyssim==0.4
from ssim.utils import get_gaussian_kernel
from skimage.measure import compare_ssim, compare_psnr


def compare(path_Groundimage, path_Generate_image):
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    size = (64, 64)

    im = Image.open(path_Groundimage)
    im = im.resize(size, Image.ANTIALIAS)

    # slightly rotated image
    im_rot = Image.open(path_Generate_image)
    im_rot = im_rot.resize(size, Image.ANTIALIAS)
    # print("========== SSIM results ==========")
    ssim_rot = SSIM(im, gaussian_kernel_1d).ssim_value(im_rot)
    # print("========== CV-SSIM results ==========")
    cw_ssim_rot = SSIM(im).cw_ssim_value(im_rot)
    # print("CW-SSIM of generated image and ground truth with  view 0 %.4f" % cw_ssim_rot)
    return ssim_rot, cw_ssim_rot


# img1 = "/home/gede/VcGAN/UESTC/Cross_view/GeneratedImage/a13_ov4_gv5_d4_p101_c2.png"
# img2 = "/home/gede/VcGAN/Dataset/Test_image/a13_v5_d5_p102_c2.png"

# img1 = "/home/gede/VcGAN/UESTC/Cross_view/GeneratedImage/a08_ov6_gv7_d6_p008_c2.png"
# img2 = "/home/gede/VcGAN/Dataset/Test_image/a20_v7_d7_p078_c2.png"

# s, c = compare(img1, img2)
# print(s)
# print(c)
image_dir = "/home/gede/VcGAN/Dataset/Test_image/"
Gen_path = "/home/gede/VcGAN/UpdateVersion/Cross_view/Evaluation /GeneratedImage/"
imageName = input("Enter the Generated Image action and view like a00_v1 : \n ")  # a09_ov4_gv7_d4_p039_c2.png
count_in = count_out = 0
sum_ssim = 0
sum_cwssim = 0
total_ssim = total_cwssim = 0
max_ssim = min_ssim = 0
max_cwssim = min_cwssim = 0
max_log_ssim, max_log_cwssim = [], []
for i in range(40):
    if i < 10:
        action = imageName[0] + '0' + str(i)
    else:
        action = imageName[0] + str(i)
    for gen in glob.glob(Gen_path + action + '_ov' + imageName[6] + '_gv' + imageName[-1] + '*.*'):
        print(gen)
        count_out = count_out + 1
        for img in glob.glob(image_dir + action + '_v' + imageName[-1] + '*.*'):
            print(img)
            ssim, cwssim = compare(img, gen)

            #  ssim min and maxmim
            if max_ssim < ssim:
                max_ssim = ssim
                max_log_cwssim.append(max_ssim)
            if max_cwssim < cwssim:
                max_cwssim = cwssim
                max_log_ssim.append(max_cwssim)
                GroundImage = img
                GeneImage = gen
            sum_ssim = sum_ssim + ssim
            sum_cwssim = sum_cwssim + cwssim
            count_in = count_in + 1
        ssim = sum_ssim / count_in
        cwssim = sum_cwssim / count_in

        print("Avg of ssim \t : " + str(ssim))
        print("Avg of cwssim\t :" + str(cwssim))

        total_ssim = total_ssim + ssim
        total_cwssim = total_cwssim + cwssim

print(" end of searching image  and  calculating final SSIM and CW-SSIm ")
ssim = total_ssim / count_out
cwssim = total_cwssim / count_out

print("========= Summery  =====================")
print('  ')
print("the ground truth image of the genrated image is: " + GeneImage)
print("the ground truth image of the genrated image is: " + GroundImage)
print("Avg of ssim \t : " + str(ssim))
print("Avg of cwssim\t :" + str(cwssim))
print("max of CW-SSIM \t " + str(max_cwssim))
print("max of SSIM \t " + str(max_ssim))
import matplotlib.pyplot as plt
plt.plot(range(1, len(max_log_cwssim)+1), max_log_cwssim,"b", label ='max_cw_ssim')
plt.show()
