
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from utils import *

# 加载.mat文件
path = '/home/s1u1/code/HSI_image_fusion/fusion_code/train_sota/SIGNet/'

path1 = os.path.join(path + 'predict/')
path2 = os.path.join(path + 'pre_rgb/')
path3 = os.path.join(path + 'rea_rgb/')
path4 = os.path.join(path + 'mis_rgb/')
path5 = os.path.join(path + 'lrhsi/')
path6 = os.path.join(path + 'mis/')
# path7 = os.path.join(path + '/final_map/')
# path8 = os.path.join(path + 'aatmap/')
# path9 = os.path.join(path + 'redesq/')
path10= os.path.join(path + 'mis_all_rgb/')
# path1 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/harvard_abi_loss_att/predict/'
# path2 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/harvard_abi_loss_att/pre_rgb/'
# path3 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/harvard_abi_loss_att/rea_rgb/'
# path4 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/harvard_abi_loss_att/mis_rgb/'
# path5 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/harvard_abi_loss_att/lrhsi/'
# path6 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/harvard_abi_loss_att/msi/'
# # path7 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/stage_base/final_map/'
# # path8 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/stage_base/aatmap/'
# # path9 = '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/stage_base/redesq/'
# path10= '/home/s1u1/code/HSI_image_fusion/fusion_code/process_model/harvard_abi_loss_att/mis_all_rgb/'
channel = 4
PSF = fspecial('gaussian', 8, 3)
downsample_factor = 8
R = create_F()


def loss_att_v3(a,b,c): #hsi,predict,GT
    at=0
    a = np.transpose(a, (2, 1, 0))
    HW = a.shape[2]*a.shape[1]
    ar=np.reshape(a,(a.shape[0],a.shape[1]*a.shape[2]))
    HW = int(HW)
    aball=np.zeros((1,b.shape[1],b.shape[2]))
    for i in range(HW):
        at+=1
        arn=ar[:,i]
        arn=arn[:, np.newaxis]
        aexp = np.tile(arn, b.shape[1]*b.shape[2])
        # aexp=arn.expand(b.shape[0],b.shape[1]*b.shape[2])
        ares=np.reshape(aexp,(b.shape[0],b.shape[1],b.shape[2]))
        abssum=np.sum(np.abs(ares-b),axis=0)
        abssumsq=abssum[np.newaxis,:,: ]
        abssumsq_g1=(abssumsq-abssumsq.min())/(abssumsq.max()-abssumsq.min()+0.0000001) #hardsubmap

        abssumsq_g1=abssumsq_g1+abssumsq_g1*(1-abssumsq_g1) #grown function

        aball=aball+abssumsq_g1
    # aatmap=(aball-aball.min())/(aball.max()-aball.min()+0.0000001)
    aatmap=aball/HW #hardmap
    # rede=np.sum(np.abs(next(iter(c))-next(iter(b))),axis=0)/b.shape[2] #recons_degree #v3+
    rede=np.abs(next(iter(c))-next(iter(b)))
    redesq=rede
    redesq_g1=(redesq-redesq.min())/(redesq.max()-redesq.min()+0.0000001) #reconsdegreemap
    final_map=aatmap*rede

    return final_map,aatmap,redesq_g1

'''颜色转灰度'''
fileList1=os.listdir(path1)
fileList1.sort(key=lambda x:x[:-4])
# fileList2=os.listdir(path2)
# image_size = 128

print('read over')
for i in fileList1:
    a =  os.path.join(path1 + os.sep + os.path.splitext(i)[0] + '.mat')
    
    image = h5py.File(a) 
    # 获取数据和维度信息
    image_data = image['fak']  # 假设数据在 key 为 'data' 的字段中
    image_data = np.transpose(image_data, (2, 1, 0))
    width, height, num_channels = image_data.shape
    sign = 3
    # 初始化平移后的图像数组
    shifted_image = np.zeros(( width+num_channels*sign,height+num_channels*sign))

    ones_image = np.ones(( width,height))
    # 将每个通道相对于前一个通道向右和向上分别平移一个像素
    for v in range(num_channels):
        c= num_channels-v
        shifted_image[num_channels*sign-c*sign : width+num_channels*sign-c*sign , c*sign : height+c*sign ] = image_data[:,:,v]

    # 归一化数据以便于可视化
    shifted_image_normalized = shifted_image
    b =  os.path.join(path2 + os.sep + os.path.splitext(i)[0] + '.png')
    plt.imsave(b,shifted_image_normalized, cmap='viridis')



    # image_data_r = image['rea']  # 假设数据在 key 为 'data' 的字段中
    # MSI = np.tensordot(R, image_data_r, axes=([1], [0]))

    # MSI = np.transpose(MSI, (2, 1, 0))*255
    # zero_msi = np.zeros_like(MSI)
    # zero_msi[:,:,2] = MSI[:,:,0]
    # zero_msi[:,:,1] = MSI[:,:,1]
    # zero_msi[:,:,0] = MSI[:,:,2]

    # b =  os.path.join(path6 + os.sep + os.path.splitext(i)[0] + '.png')
    # cv2.imwrite(b, zero_msi)


    
    image_data_r = image['rea']  # 假设数据在 key 为 'data' 的字段中
    HSI_LR = Gaussian_downsample(image_data_r, PSF, downsample_factor)
    HSI_LR = np.transpose(HSI_LR, (2, 1, 0))
    width, height, num_channels = HSI_LR.shape
    sign = 3
    # 初始化平移后的图像数组
    shifted_image = np.zeros(( width+num_channels*sign,height+num_channels*sign))
    mask_img = np.ones(( width+num_channels*sign,height+num_channels*sign))
    mask = np.zeros(( width,height))
    # 将每个通道相对于前一个通道向右和向上分别平移一个像素
    for v in range(num_channels):
        c= num_channels-v
        shifted_image[num_channels*sign-c*sign : width+num_channels*sign-c*sign , c*sign : height+c*sign ] = HSI_LR[:,:,v]
        mask_img[num_channels*sign-c*sign : width+num_channels*sign-c*sign , c*sign : height+c*sign ] = mask
    # 归一化数据以便于可视化
    shifted_image_normalized = shifted_image
    b =  os.path.join(path5 + os.sep + os.path.splitext(i)[0] + '.png')
    plt.imsave(b,shifted_image_normalized, cmap='viridis')
    b =  os.path.join(path5 + os.sep + 'mask_img.png')
    plt.imsave(b,mask_img, cmap='Greys')
    # shifted_image = np.zeros(( width,height))


    # final_map,aatmap,redesq = loss_att_v3(HSI_LR,image['rea'],image['fak'])
    # b =  os.path.join(path7 + os.sep + os.path.splitext(i)[0] + '.png')
    # final_map = np.transpose(final_map, (1, 2, 0))
    # cv2.imwrite(b, final_map*255)
    # b =  os.path.join(path8 + os.sep + os.path.splitext(i)[0] + '.png')
    # aatmap = np.transpose(aatmap, (1, 2, 0))
    # cv2.imwrite(b, aatmap*255)
    # b =  os.path.join(path9 + os.sep + os.path.splitext(i)[0] + '.png')
    # cv2.imwrite(b, redesq*255)



    image_data_r = image['rea']  # 假设数据在 key 为 'data' 的字段中
    image_data_r = np.transpose(image_data_r, (2, 1, 0))
    width, height, num_channels = image_data_r.shape
    sign = 3
    # 初始化平移后的图像数组
    shifted_image = np.zeros(( width+num_channels*sign,height+num_channels*sign))
    mask_img = np.ones(( width+num_channels*sign,height+num_channels*sign))
    mask = np.zeros(( width,height))
    # 将每个通道相对于前一个通道向右和向上分别平移一个像素
    for v in range(num_channels):
        c= num_channels-v
        shifted_image[num_channels*sign-c*sign : width+num_channels*sign-c*sign , c*sign : height+c*sign ] = image_data_r[:,:,v]
        mask_img[num_channels*sign-c*sign : width+num_channels*sign-c*sign , c*sign : height+c*sign ] = mask
    # 归一化数据以便于可视化
    shifted_image_normalized = shifted_image
    b =  os.path.join(path3 + os.sep + os.path.splitext(i)[0] + '.png')
    plt.imsave(b,shifted_image_normalized, cmap='viridis')
    b =  os.path.join(path3 + os.sep + 'mask_img.png')
    plt.imsave(b,mask_img, cmap='Greys')
    # shifted_image = np.zeros(( width,height))


    absimg_temp = np.zeros(( width,height))
    for v in range(num_channels):
        absimg = np.abs(image_data_r[:,:,v]-image_data[:,:,v])
        absimg_temp += absimg
    absimg_temp=absimg_temp/num_channels
    b =  os.path.join(path10 + os.sep + os.path.splitext(i)[0] + '.png')
    plt.imsave(b,absimg_temp, cmap='jet')

    absimg = np.abs(image_data_r[:,:,channel]-image_data[:,:,channel])
    b =  os.path.join(path4 + os.sep + os.path.splitext(i)[0] + '.png')
    plt.imsave(b,absimg, cmap='jet')

    
     
print('write over')
