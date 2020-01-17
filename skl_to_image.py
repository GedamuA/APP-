import glob
import os
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
import math
import matplotlib as mpl
mpl.use('Agg')
import torch.nn.functional as F
import torch.distributions
import pdb

# data path
npy_path = "/home/fesian/AI_workspace/datasets/HRI40/raw_npy/"
# get rotate skeleton data name
x_view_group_loading_list = '/home/fesian/AI_workspace/datasets/HRI40/loading_list/x_view_group.npy'
x_view_group = np.load(x_view_group_loading_list).item()
x_view_group_name = x_view_group['test']

# image save list
save_path = "/home/gede/VcGAN/Dataset/Test_image/"

# save skeleton data as image
def Save_image(data,Image_path):
    size = data.shape[0]
    mat = data.reshape((size, 25, 3))
    skeleton = np.expand_dims(mat, axis=0)
    train_mats = torch.FloatTensor(skeleton).cuda()
    input_mats = train_mats.permute(0, 3, 1, 2).contiguous()[:, :, :, :].float()
    RGB_image = nn.functional.interpolate(input_mats,
                                          size=(128, 128),
                                          mode='bilinear',
                                          align_corners=False)
    R_image = RGB_image[0].cpu()
    image_train = R_image
    G_image = transforms.ToPILImage()(image_train).convert('RGB')
    G_image.save(Image_path)


count = 0
h = []
for name in x_view_group_name:
    # # make sure new image name
    print(npy_path + name)
    Data = np.load(npy_path + name).item()

    # get view number from npy data,input npy data
    # all view are 0,2,4,6
    view_number = Data['view']
    m = int(view_number/2)

    # get skeleton data
    data = Data['mat']

    # save original image
    # r_pic_name = name[0:4] + 'v' + str(view_number) + name[6:-4] + ".png"

    r_pic_name = name[:-14] + "v" + str(view_number) + '_' + name[-14:-4] + ".png"
    r_pic_path_name = "/home/gede/VcGAN/Dataset/Test_image/" + r_pic_name
    # save image
    Save_image(data, r_pic_path_name)


    count += 1
print(set(h))
print(count)
print('end')