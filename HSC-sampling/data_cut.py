# 分割并保存
import argparse

import numpy as np
import glob
import os
import hdf5storage as h5
from numpy import *


# cave数据集的模型函数的光谱下采样函数
from scipy import signal


def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F

def fspecial(func_name, kernel_size, sigma):
    if func_name == 'gaussian':
        m = n = (kernel_size - 1.) / 2.
        y, x = ogrid[-m:m + 1, -n:n + 1]
        h = exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


def Gaussian_downsample(x, psf, s):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y


# def normalize(data, max_val, min_val):
#     return (data - min_val) / (max_val - min_val)


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


# R是光谱下采样函数
def process_train_data(R, PSF,downsample_factor,patch_size, stride, original_path, save_path, dataset):
    if dataset == 'cave':
        key_name = 'b'
    if dataset == 'harvard':
        key_name = "ref"

    save_data_path = save_path
    print("\nprocess training set ...\n")
    patch_num = 1
    filenames_hyper = glob.glob(os.path.join(original_path, '*.mat'))
    filenames_hyper.sort()
    # for k in range(1):  # make small dataset
    for k in range(len(filenames_hyper)):
        print(filenames_hyper[k])  # 输出mat文件名字
        # load hyperspectral image
        mat = h5.loadmat(filenames_hyper[k])
        hyper = np.float32(np.array(mat[key_name]))
        hyper = np.transpose(hyper, [2, 0, 1])
        if dataset == 'harvard':  # cave 已经被处理过了
            hyper = hyper / hyper.max()

        # load rgb image
        rgb = np.tensordot(R, hyper, axes=([1], [0]))  # 得到rgb  整张的图
        HSI_LR = Gaussian_downsample(hyper, PSF, downsample_factor)   # 得到整张lr_hsi

        #h5.savemat(save_data_path, {'rgb': rgb}, format='2.3')

        patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
        patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)
        patches_lr = Im2Patch(HSI_LR, win=patch_size//downsample_factor, stride=stride//downsample_factor)

        # add data ：重组patches
        for j in range(patches_hyper.shape[3]):
            #print("generate training sample #%d" % patch_num)
            sub_hyper = patches_hyper[:, :, :, j]
            sub_rgb = patches_rgb[:, :, :, j]
            sub_lr = patches_lr[:, :, :, j]
            save_data_path_array = save_data_path
            save_path = os.path.join(save_data_path_array, str(patch_num) + '.mat')
            #print(save_data_path)
            h5.savemat(save_path, {'hr': sub_hyper}, format='2.3')
            h5.savemat(save_path, {'rgb': sub_rgb}, format='2.3')
            h5.savemat(save_path, {'lr': sub_lr}, format='2.3')
            patch_num += 1
            print("\ntraining set: # samples %d\n" % (patch_num - 1))

def process_test_data(R, PSF,downsample_factor,original_path, save_path, dataset):
    if dataset == 'cave':
        key_name = 'b'
    if dataset == 'harvard':
        key_name = "ref"

    save_data_path = save_path
    print("\nprocess training set ...\n")
    patch_num = 1
    filenames_hyper = glob.glob(os.path.join(original_path, '*.mat'))
    filenames_hyper.sort()
    # for k in range(1):  # make small dataset
    for k in range(len(filenames_hyper)):
        print(filenames_hyper[k])  # 输出mat文件名字
        # load hyperspectral image
        mat = h5.loadmat(filenames_hyper[k])
        hyper = np.float32(np.array(mat[key_name]))
        hyper = np.transpose(hyper, [2, 0, 1])
        if dataset == 'harvard':  # cave 已经被处理过了
            hyper = hyper / hyper.max()

        # load rgb image
        rgb = np.tensordot(R, hyper, axes=([1], [0]))  # 得到rgb  整张的图
        HSI_LR = Gaussian_downsample(hyper, PSF, downsample_factor)   # 得到整张lr_hsi

        #h5.savemat(save_data_path, {'rgb': rgb}, format='2.3')

        # add data ：重组patches
        save_data_path_array = save_data_path
        matname=filenames_hyper[k].replace(original_path+'\\','')
        save_path = os.path.join(save_data_path_array, matname)
        h5.savemat(save_path, {'hr': hyper}, format='2.3')
        h5.savemat(save_path, {'rgb': rgb}, format='2.3')
        h5.savemat(save_path, {'lr': HSI_LR}, format='2.3')
        patch_num += 1
        print("\ntraining set: # samples %d\n" % (patch_num - 1))

def main():

    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    downsample_factor=8
    patch_size=64
    stride=32

    ori_train_data_path=  'D:\data\cave\cave_train'
    ori_test_data_path= 'D:\data\cave\cave_test'

    save_train_data_path='D:\patch_data\cave_train'
    save_test_data_path='D:\patch_data\cave_test'

    dataset='cave'
    process_train_data(R=R, PSF=PSF,downsample_factor=downsample_factor,patch_size=patch_size, stride=stride, original_path=ori_train_data_path,
                  save_path=save_train_data_path, dataset=dataset)
    process_test_data(R=R, PSF=PSF,downsample_factor=downsample_factor,original_path=ori_test_data_path,
                 save_path=save_test_data_path, dataset=dataset)

if __name__ == '__main__':
    main()
