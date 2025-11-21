# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:41:48 2020

@author: Dian
"""
import time

import numpy as np

import torch
from scipy.io import loadmat

import os

from torch import nn

from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR
# from net_sota.Fuformer import MainNet
from net2.multiscale_specconvfront2_reschangemarx import allnet  # multiscale_base  multiscale_abi  multistage_scale  multiscale  multistage2_scale
# from net_sota.U2Net import U2Net
# from net_sota.SSTF_unet import SSTF_Unet
# from net_sota.HSRnet import RGBNet
# from net_sota.Fuformer import MainNet
# from net_sota.DSPNet import DSPNet
# from net_sota.DHIF import HSI_Fusion
# from net_sota.PSRT import PSRTnet
from utils import create_F, Gaussian_downsample, fspecial, AverageMeter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import hdf5storage

import hdf5storage as h5
# dataset = 'Harvard'
# path = '/home/s7u0/ljy/s1u1/dataset/harvard/harvard_test/'

# dataset = 'CAVE'
# path='/home/s7u0/ljy/s1u1/dataset/cave/test/' #

dataset='NTIRE2022'
path = '/home/s7u0/ljy/s1u1/dataset/NTIRE2022/Valid_spectral/'




imglist = os.listdir(path)

model_path = r'/home/s7u0/ljy/s1u1/code/HSI_image_fusion/fusion_code/process_model2/ntire_multiscale_specconvfront2_reschangemarx/2/NTIRE2022_pkl/500EPOCH_RMSE_best.pkl'
R = create_F()
R_inv = np.linalg.pinv(R)
R_inv = torch.Tensor(R_inv)
R = torch.Tensor(R)

# net2 = VSR_CAS(channel0=31, factor=8, P=torch.Tensor(R)).cuda()
net2=allnet(48).cuda()
# net2 = PSRTnet(1).cuda()
# net2 = U2Net(64).cuda()
# net2 = SSTF_Unet().cuda()
# net2 = RGBNet(31,6,64).cuda()#(150,6,256).cuda()
# net2 = MainNet().cuda()
# net2 = DSPNet(31, 3).cuda()
# net2 = HSI_Fusion(Ch=31, stages=4, sf=8).cuda()
# net2=MainNet().cuda()
# net2 = RGBNet().cuda()

checkpoint = torch.load(model_path)  # 加载断点
# net2.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
net2.load_state_dict(checkpoint)
save_path = '/home/s7u0/ljy/s1u1/code/HSI_image_fusion/fusion_code/test_save/multiscale_specconvfront2_reschangemarx_ntire/mat/'

RMSE = []
training_size = 64
stride = 32
PSF = fspecial('gaussian', 8, 3)
downsample_factor = 8

loss_func = nn.L1Loss(reduction='mean').cuda()


def reconstruction(net2, R, HSI_LR, MSI, HRHSI, downsample_factor, training_size, stride, val_loss):
    index_matrix = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            temp_hrhs = HRHSI[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                out,_,_ = net2(temp_lrhs,temp_hrms)
                assert torch.isnan(out).sum() == 0

                loss_temp = loss_func(out, temp_hrhs.cuda())
                val_loss.update(loss_temp)
                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon, val_loss


val_loss = AverageMeter()
SAM = Loss_SAM()
RMSE = Loss_RMSE()
PSNR = Loss_PSNR()
sam = AverageMeter()
rmse = AverageMeter()
psnr = AverageMeter()
time_mean = 0
for i in range(0, len(imglist)):
    net2.eval()
    # img = loadmat(path + imglist[i])
    if dataset=='CAVE':
        img = loadmat(path + imglist[i])
        img1 = img["b"]
        # img1=img1/img1.max()
    elif dataset=='Harvard':
        img = loadmat(path + imglist[i])
        img1 = img["ref"]
        img1=img1/img1.max()
    elif dataset=='NTIRE2022':
        img = h5.loadmat(path + imglist[i])
        img1 = img["cube"]
        img1 = img1 / img1.max()
    # print("real_hyper's shape =",img1.shape)

    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
    if dataset=='NTIRE2022':
        HRHSI = np.transpose(img1, (2, 0, 1))
        HRHSI = np.pad(HRHSI, ((0, 0), (3, 3), (0, 0)), mode='reflect')
        HRHSI = torch.Tensor(HRHSI)
    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
    MSI_1 = torch.unsqueeze(MSI, 0)
    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
    time1 = time.time()
    to_fet_loss_hr_hsi = torch.unsqueeze(torch.Tensor(HRHSI), 0)

    with torch.no_grad():
        prediction, val_loss = reconstruction(net2, R, HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi,
                                              downsample_factor, training_size, stride, val_loss)

        Fuse = prediction.cpu().detach().numpy()

    time2 = time.time()
    print(time2 - time1)
    timea=time2 - time1
    time_mean=time_mean+timea
    sam.update(SAM(np.transpose(HRHSI.cpu().detach().numpy(), (1, 2, 0)),
                   np.transpose(prediction.squeeze().cpu().detach().numpy(), (1, 2, 0))))
    rmse.update(RMSE(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
    psnr.update(PSNR(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))

    faker_hyper = np.transpose(Fuse, (1, 2, 0))
    print(i, ':', imglist[i],faker_hyper.shape)
    print(PSNR(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))

    test_data_path = os.path.join(save_path + imglist[i])
    hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
    hdf5storage.savemat(test_data_path, {'rea': img1}, format='7.3')
time_mean=time_mean/len(imglist)
print(len(imglist))
print("time_mean:",time_mean)
print("val loss:",val_loss.avg)
print("val  PSNR:", psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg)
