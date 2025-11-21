import glob
import os

import hdf5storage as h5
from torch.utils import data
import utils
from utils import *
import numpy as np
import torch
from torch.utils.data import Dataset
# import torch.nn.functional as F


class RealDATAProcess2(Dataset):
    def __init__(self, hsi,msi, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        # hwc-chw
        HRHSI = np.transpose(hsi, (2, 0, 1))
        msi = np.transpose(msi, (2, 0, 1))

        HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
        # MSI = Gaussian_downsample(msi, PSF, downsample_factor)
        MSI = msi

        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                if (j+training_size)>800 and k<400:
                    pass
                else:
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    temp_hrms = temp_hrms.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_lrhs.append(temp_lrhs)
                    train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]

class RealDATAProcess(Dataset):
    def __init__(self, hsi,msi, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        HRHSI=hsi
        # hwc-chw
        HSI_LR = Gaussian_downsample(hsi, PSF, downsample_factor)
        MSI = Gaussian_downsample(msi, PSF, downsample_factor)

        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs=temp_hrhs.astype(np.float32)
                temp_lrhs=temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]



class CAVEHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["b"]
            # img1 = img["ref"]
            # img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class NTIREHSIMosaicDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, msfa_size, spatial_ratio, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)

        train_target, train_mosaic, train_pan = [], [], []
        

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["cube"]
            # img1 = img["ref"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            HRHSI = np.pad(HRHSI, ((0, 0), (15, 15), (0, 0)), mode='reflect')
            hrms = HRHSI
            # hrms_select_bands = hrms[:hrms.shape[0]//(msfa_size*spatial_ratio)*(msfa_size*spatial_ratio),
            #                             :hrms.shape[1]//(msfa_size*spatial_ratio)*(msfa_size*spatial_ratio),
            #                             12:28].astype(numpy.float32)
            hrms_select_bands = hrms[12:28,:hrms.shape[1]//(msfa_size*spatial_ratio)*(msfa_size*spatial_ratio),
                                        :hrms.shape[2]//(msfa_size*spatial_ratio)*(msfa_size*spatial_ratio)
                                        ].astype(numpy.float32)

            MSFA = numpy.array([[0, 1, 2, 3],
                                [4, 5, 6, 7],
                                [8, 9, 10, 11],
                                [12, 13, 14, 15]])
            hrms_tensor = torch.from_numpy(hrms_select_bands).unsqueeze(0)#.permute(2, 0, 1).unsqueeze(0)
            # lrms_tensor = torch.nn.functional.avg_pool2d(hrms_tensor, 2, 2)
            hrms = hrms_tensor[0].permute(1, 2, 0).numpy()
            mosaic = utils.MSFA_filter(hrms, MSFA)
            spe_res = numpy.array([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1])
            spe_res /= spe_res.sum()
            hrms_selected_bands = np.transpose(hrms_select_bands, (1, 2, 0))
            pan = numpy.sum(hrms_selected_bands * spe_res, axis=-1, keepdims=True)
            mosaic = np.transpose(mosaic, (2, 0, 1))
            # mosaic = mosaic / mosaic.max()
            # HRHSI = np.transpose(HRHSI, (2, 0, 1))
            pan = np.transpose(pan, (2, 0, 1))
            # pan = pan / pan.max()
            for j in range(0, hrms_select_bands.shape[1] - training_size + 1, stride):
                for k in range(0, hrms_select_bands.shape[2] - training_size + 1, stride):
                    temp_hrhs = hrms_select_bands[:, j:j + training_size, k:k + training_size]
                    temp_mosaic = mosaic[:, j:j + training_size, k:k + training_size]
                    temp_pan = pan[:, j:j + training_size, k:k + training_size]
                    # temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                    #             int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_mosaic=temp_mosaic.astype(np.float32)
                    temp_pan=temp_pan.astype(np.float32)
                    train_target.append(temp_hrhs)
                    train_mosaic.append(temp_mosaic)
                    train_pan.append(temp_pan)

#                 pan = numpy.sum(hrms_select_bands * spe_res, axis=-1, keepdims=True)
#                 
#                 spatial_ratio = pan.shape[0] // mosaic.shape[0]
#                 if type == "train":
#                     mosaic_patches = crop_to_patch(mosaic, args.train_size//spatial_ratio, args.stride//spatial_ratio)
#                     pan_patches = crop_to_patch(pan, args.train_size, args.stride)

#                     self.mosaic += mosaic_patches
#                     self.pan += pan_patches

#                 elif type == "test":
#                     self.mosaic.append(mosaic)
#                     self.pan.append(pan)
#                     self.hrms.append(hrms_select_bands)
            # mosaic = np.transpose(mosaic, (2, 0, 1))
            # # HRHSI = np.transpose(HRHSI, (2, 0, 1))
            # pan = np.transpose(pan, (2, 0, 1))
            # self.mosaic.append(mosaic)
            # self.target.append(HRHSI)
            # self.pan1.append(pan)
        train_mosaic = torch.Tensor(np.array(train_mosaic))
        train_target = torch.Tensor(np.array(train_target))
        train_pan = torch.Tensor(np.array(train_pan))
        self.train_mosaic_all = train_mosaic
        self.train_target_all = train_target
        self.train_pan_all = train_pan


    def __getitem__(self, index):
        train_mosaic = self.train_mosaic_all[index, :, :, :]
        train_target = self.train_target_all[index, :, :, :]
        train_pan = self.train_pan_all[index, :, :, :]

        return train_target, train_mosaic, train_pan

    def __len__(self):
        return self.train_mosaic_all.shape[0]
    #         # hwc-chw
    #         HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
    #         MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
    #         for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
    #             for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
    #                 temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
    #                 temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
    #                 temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
    #                             int(k / downsample_factor):int((k + training_size) / downsample_factor)]
    #                 # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
    #                 temp_hrhs=temp_hrhs.astype(np.float32)
    #                 temp_hrms=temp_hrms.astype(np.float32)
    #                 temp_lrhs=temp_lrhs.astype(np.float32)
    #                 train_hrhs.append(temp_hrhs)
    #                 train_hrms.append(temp_hrms)
    #                 train_lrhs.append(temp_lrhs)
    #     train_hrhs = torch.Tensor(np.array(train_hrhs))
    #     train_hrms = torch.Tensor(np.array(train_hrms))
    #     train_lrhs = torch.Tensor(np.array(train_lrhs))

    #     # print(train_hrhs.shape, train_hrms.shape)
    #     self.train_hrhs_all = train_hrhs
    #     self.train_hrms_all = train_hrms
    #     self.train_lrhs_all = train_lrhs

    # def __getitem__(self, index):
    #     train_hrhs = self.train_hrhs_all[index, :, :, :]
    #     train_hrms = self.train_hrms_all[index, :, :, :]
    #     train_lrhs = self.train_lrhs_all[index, :, :, :]
    #     # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
    #     return train_hrhs, train_hrms, train_lrhs

    # def __len__(self):
    #     return self.train_hrhs_all.shape[0]

class NTIREHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["cube"]
            # img1 = img["ref"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            HRHSI = np.pad(HRHSI, ((0, 0), (3, 3), (0, 0)), mode='reflect')
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]

class HarvardHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["ref"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class RealDATAProcess3(Dataset):
    def __init__(self, LR,msi,HR, training_size, stride, downsample_factor):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        HSI_LR = LR
        MSI = msi
        HRHSI=HR
        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                # if (j+training_size)>800 and k<400:
                #     pass
                # else:
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs=temp_hrhs.astype(np.float32)
                temp_lrhs=temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.sha
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]