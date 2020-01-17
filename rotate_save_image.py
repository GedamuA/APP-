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

# data path
npy_path = "/home/fesian/AI_workspace/datasets/HRI40/raw_npy/"
# get rotate skeleton data name
x_view_group_loading_list = '/home/fesian/AI_workspace/datasets/HRI40/loading_list/x_view_group.npy'
x_view_group = np.load(x_view_group_loading_list).item()
x_view_group_name = x_view_group['train']

# image save list
save_path = "/home/zhanghao/skeleton_data/rotate_img/"

# view list
list_i = ["1", '3', "5", '7']

# rotate angle[x,y,z]
angles1 = [45, 0, 0]


# rotate function,input angle and skeleton data get transfered data
def AnglesToRotationMatrix(angles1, skeleton):
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0] * 3.141592653589793 / 180.0
    theta[1] = angles1[1] * 3.141592653589793 / 180.0
    theta[2] = angles1[2] * 3.141592653589793 / 180.0
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), math.sin(theta[0])],
                    [0, -math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), math.sin(theta[1]), 0],
                    [-math.sin(theta[1]), math.cos(theta[1]), 0],
                    [0, 0, 1]
                    ])
    R_z = np.array([[math.cos(theta[2]), 0, -math.sin(theta[2])],
                    [0, 1, 0],
                    [math.sin(theta[2]), 0, math.cos(theta[2])]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))

    size = skeleton.shape[0]
    mat = skeleton.reshape((size, 25, 3))
    data = np.dot(mat, R)
    return data


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

# rotate 2-3;4-5;6-7 with degree 45
count = 0
h = []
for name in x_view_group_name:
    # # make sure new image name
    Data = np.load(npy_path + name).item()

    # get view number from npy data,input npy data
    # all view are 0,2,4,6
    view_number = Data['view']
    m = int(view_number/2)

    # get skeleton data
    data = Data['mat']

    # save original image
    r_pic_name = name[:-4] + "_v" + str(view_number) + ".png"
    r_pic_path_name = "/home/zhanghao/skeleton_data/raw_image/" + r_pic_name
    # save image
    Save_image(data, r_pic_path_name)

    # # rotate image,then save
    # tfs_data = AnglesToRotationMatrix(angles1, data)
    # pic_name = name[:-4]+"_v" + list_i[m]+".png"
    # pic_path_name = save_path + pic_name
    # # save image
    # Save_image(tfs_data, pic_path_name)


    count += 1
print(set(h))
print(count)
print('end')