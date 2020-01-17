import os
from random import choice
import shutil
import pdb
imgs = []
orginalPath = "/home/gede/VcGAN/Dataset/Test_image/"
forTrain = "/home/gede/VcGAN/Dataset/ForTrain/"
forTest = "/home/gede/VcGAN/Dataset/ForTest/"


for (dirname, dirs,files) in os.walk(orginalPath):
    for filename in files:
        imgs.append(filename)

countforTrain = int(len(imgs) * 0.5)
countforTest = int(len(imgs) * 0.5)

for train in range(countforTrain):
    images = choice(imgs)

    shutil.move(os.path.join(orginalPath, images), os.path.join(forTrain, images))
    imgs.remove(images)

for test in range(countforTest):
    images = choice(imgs)

    shutil.move(os.path.join(orginalPath, images), os.path.join(forTest, images))
    imgs.remove(images)
print(len(os.listdir(orginalPath)))
print(len(os.listdir(forTrain)))
print(len(os.listdir(forTest)))

