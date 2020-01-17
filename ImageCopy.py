import glob
import os
import shutil

import pdb

main_path = "/home/gede/VcGAN/Dataset/Test_image/"
new_path = "/home/gede/VcGAN/Dataset/TestImageV7/"
imageName = input("Enter the Generated Image action and view like a00_v1 : \n ") # a09_ov4_gv7_d4_p039_c2.png
count =0
for i in range(40):
    if i < 10:
        action = imageName[0] + '0' + str(i)
    else:
        action = imageName[0] + str(i)
    for img in glob.glob(main_path + action + '_v7' + '*.*', recursive=True):
        print(img)
        shutil.copy(img, new_path)
