
import glob
import os
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms

import matplotlib as mpl
mpl.use('Agg')
import torch.nn.functional as F
import torch.distributions

npy_path = "/home/fesian/AI_workspace/datasets/HRI40/raw_npy/"

# get circle estimate skeleton data
file_name = glob.glob(npy_path+"*d8*.npy")
print(len(file_name))

save_path = "/home/zhanghao/skeleton_data/npy/"
# save_path = "/home/zhanghao/temp/"
# transfer skeleton data to image
list_i = ["0", '1', '2', "3", '4', '5', "6",
          '7']


def _down_sampling(mat, nclips):
    samples_num = mat.shape[0]
    each_clip_size = int(samples_num / nclips)
    index_list = []
    begin = 0
    end = 0
    for each in range(nclips):
        end += each_clip_size
        index_list.append((begin, end))
        begin = end
    random_list = []
    for each_index in index_list:
        random_id = random.sample(list(range(each_index[0], each_index[1])), 1)[0]
        random_list.append(random_id)
    sample_mat = mat[random_list]
    return sample_mat


def _zero_padding(sample, nclips):
    retsample = np.zeros((nclips, 25, 3))
    length = sample.shape[0]
    retsample[:length, :, :] = sample
    return retsample


def _length_norm(mat):
    '''
    larger than gc.norm_length will clip equally and smaller than gc.norm_length will expand to 2 times and clip
    '''
    size = mat.shape[0]
    mat = mat.reshape((size, 25, 3))
    if size > 100:
        mat = _down_sampling(mat, 100)
    else:
        mat = _zero_padding(mat, 100)
    return mat


for name in file_name:
    skeleton = np.load(name).item()
    npy_name = os.path.basename(name)[:-4]
    save_name = save_path + npy_name

    # split all frames to 16 parts
    skeleton_data = skeleton['mat']
    frames = skeleton_data.shape[0]
    one_copy = frames//16
    for i in range(15):
        m = i*one_copy
        data = skeleton_data[m:m+one_copy]
        skeleton = _length_norm(data)

        # npy data
        new_path = save_name+"_"+list_i[i]+".npy"
        np.save(new_path, skeleton)

        # image
        # img_path = save_name + "_" + list_i[i] + ".jpg"
        # skeleton = np.expand_dims(skeleton, axis=0)
        # train_mats = torch.FloatTensor(skeleton).cuda()
        # input_mats = train_mats.permute(0, 3, 1, 2).contiguous()[:, :, :, :].float()
        # RGB_image = nn.functional.interpolate(input_mats,
        #                                       size=(128, 128),
        #                                       mode='bilinear',
        #                                       align_corners=False)
        # R_image = RGB_image[0].cpu()
        # image_train = R_image
        # G_image = transforms.ToPILImage()(image_train).convert('RGB')
        # G_image.save(img_path)

    m = 15 * one_copy
    data = skeleton_data[m:frames-1]
    skeleton = _length_norm(data)

    # data
    # print(data.shape)
    new_path = save_name + "_" + list_i[15] + ".npy"
    np.save(new_path, skeleton)

    # # image
    # img_path = save_name + "_" + list_i[i] + ".png"
    #
    # skeleton = np.expand_dims(skeleton, axis=0)
    # train_mats = torch.FloatTensor(skeleton).cuda()
    # input_mats = train_mats.permute(0, 3, 1, 2).contiguous()[:, :, :, :].float()
    # RGB_image = nn.functional.interpolate(input_mats,
    #                                       size=(128, 128),
    #                                       mode='bilinear',
    #                                       align_corners=False)
    # R_image = RGB_image[0].cpu()
    # image_train = R_image
    # G_image = transforms.ToPILImage()(image_train).convert('RGB')
    # G_image.save(img_path)

print("end")