import glob, os
import random

ForTrain = "/home/gede/VcGAN/Dataset/ForTrain/"
Train = "/home/gede/VcGAN/Dataset/Train_image/a12_v0_d4_p044_c1.png"
views = ["1", "3", "5", "7"]


def read_image(imagName):
    for img in glob.glob(ForTrain + imagName + "*.*"):
        if img is None:
            break
    return img


def get_image(TrainPath):
    tmp = random.randint(0, 3)
    v = views[tmp]
    token = TrainPath.split("/")
    img2 = token[6][0:5] + str(v) + '_d' + str(v)
    img2 = read_image(img2)
    return img2


print("Train_image ==>" + str(Train))
print("Generated Image==> " + str(get_image(Train)))
