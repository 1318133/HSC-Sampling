import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
# import torch_dct as DCT
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y= self.dropout(y)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.dropout = nn.Dropout(p=0.1)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        x_out=self.dropout(x_out)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class Sobel_operation_c(nn.Module):
    def __init__(self, channel):
        super(Sobel_operation_c, self).__init__()
        # 定义Sobel算子
        self.sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        # 转换为卷积层的权重
        self.sobel_xx = nn.Parameter(self.sobel_x).cuda()
        self.sobel_yy = nn.Parameter(self.sobel_y).cuda()
        
    def forward(self, x):
        for i in range(x.size(1)):
            feat_grad_x = F.conv2d(x[:, i:i + 1, :, :], self.sobel_xx, stride=1, padding=1)
            feat_grad_y = F.conv2d(x[:, i:i + 1, :, :], self.sobel_yy, stride=1, padding=1)
            edges = torch.sqrt(feat_grad_x**2 + feat_grad_y**2)
            if i == 0:
                edges_s = edges
            else:
                edges_s = torch.cat((edges_s, edges), dim=1)
        return edges_s

def Sobel_operation(x):

    # 定义Sobel算子
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    # 转换为卷积层的权重
    sobel_x = nn.Parameter(sobel_x).cuda()
    sobel_y = nn.Parameter(sobel_y).cuda()
    for i in range(x.size(1)):
        feat_grad_x = F.conv2d(x[:, i:i + 1, :, :], sobel_x, stride=1, padding=1)
        feat_grad_y = F.conv2d(x[:, i:i + 1, :, :], sobel_y, stride=1, padding=1)
        edges = torch.sqrt(feat_grad_x**2 + feat_grad_y**2)
        if i == 0:
            edges_s = edges
        else:
            edges_s = torch.cat((edges_s, edges), dim=1)

    # # 计算x和y方向的边缘
    # edge_x = F.conv2d(x, sobel_x)
    # edge_y = F.conv2d(x, sobel_y)
    # # 计算总边缘
    # edges = torch.sqrt(edge_x**2 + edge_y**2)
    return edges

def qiancha(A,B,C,D):
    # 获取通道数
    num_channels = A.size(1)  # 假设 A, B, C, D 的通道数相同
    b,c,h,w = A.size()
    # 创建一个新的张量来存储结果，尺寸为 (32, 31*4, 64, 64)
    result = torch.empty(b, c * 4, h, w, device=A.device)

    # 使用切片直接赋值
    result[:, 0::4, :, :] = A  # 插入 A 的所有通道
    result[:, 1::4, :, :] = B  # 插入 B 的所有通道
    result[:, 2::4, :, :] = C  # 插入 C 的所有通道
    result[:, 3::4, :, :] = D  # 插入 D 的所有通道

    # # 按照通道的规律进行插入
    # for c in range(num_channels):  # 遍历每个通道
    #     result[:, c * 4, :, :] = A[:, c, :, :]  # 插入 A 的 c 通道
    #     result[:, c * 4 + 1, :, :] = B[:, c, :, :]  # 插入 B 的 c 通道
    #     result[:, c * 4 + 2, :, :] = C[:, c, :, :]  # 插入 C 的 c 通道
    #     result[:, c * 4 + 3, :, :] = D[:, c, :, :]  # 插入 D 的 c 通道

    return result

def qiancha2(A,B):
    # 获取通道数
    num_channels = A.size(1)  # 假设 A, B, C, D 的通道数相同
    b,c,h,w = A.size()
    # 创建一个新的张量来存储结果，尺寸为 (32, 31*4, 64, 64)
    result = torch.empty(b, c * 2, h, w, device=A.device)

    # 使用 torch.cat 沿通道维度拼接
    result = torch.empty(b, c * 2, h, w, device=A.device)  # 创建结果张量
    result[:, 0::2, :, :] = A  # 插入 A 的通道
    result[:, 1::2, :, :] = B  # 插入 B 的通道

    # # 按照通道的规律进行插入
    # for c in range(num_channels):  # 遍历每个通道
    #     result[:, c * 2, :, :] = A[:, c, :, :]  # 插入 A 的 c 通道
    #     result[:, c * 2 + 1, :, :] = B[:, c, :, :]  # 插入 B 的 c 通道

    return result




class ResBlock_changemarx(nn.Module):
    def __init__(self, channel,T_channel):
        super(ResBlock_changemarx, self).__init__()
        self.channel=channel
        self.T_channel=T_channel
        self.dropout = nn.Dropout(p=0.5)
        self.ps = nn.PixelShuffle(2)
        self.layers = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                    )
        self.HWC = nn.Conv2d(self.channel, self.channel*3, kernel_size=1)
        self.conv_ps_down2 = nn.Conv2d(self.channel*2, self.channel, kernel_size=2, stride=2)
        self.HWC_dwconv = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, stride=1, padding=1, groups=self.channel*3)
        # self.HWC1 = nn.Conv2d(self.channel, self.channel*3, kernel_size=1)
        # self.HWC_dwconv1 = nn.Conv2d(self.channel*3, self.channel*3, kernel_size=3, stride=1, padding=1, groups=self.channel*3)
        self.Sobel_conv = nn.Sequential(nn.Conv2d(self.channel*3, self.channel, 1, stride=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                    )
        # self.sobel_c = Sobel_operation_c(1)
        # self.cat_conv = nn.Sequential(nn.Conv2d(self.channel*2, self.channel, 1, stride=1),
        #                             nn.LeakyReLU(negative_slope=0.1, inplace=True)
        #                             )
        self.conv_CW = nn.Sequential(nn.Conv2d(self.T_channel, self.T_channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                    )
        self.conv_HC = nn.Sequential(nn.Conv2d(self.T_channel, self.T_channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                    )
        self.conv_HW = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                    )
    def forward(self, x):
        HWC = self.HWC_dwconv(self.HWC(x))
        HW,CW,HC = HWC.chunk(3, dim=1) 
        # HWC1 = self.HWC_dwconv1(self.HWC1(x))
        # HW1,CW1,HC1 = HWC.chunk(3, dim=1) 
        x_sobel_HW = self.conv_HW(x)#Sobel_operation(x)
        x_sobel_HW_a = x_sobel_HW*HW
        x_sobel_CW = self.conv_CW(x.permute(0, 2, 1, 3))#Sobel_operation(x.permute(0, 2, 1, 3))  # 将特征张量转置为目标形状 (B, H, C, W)
        x_sobel_CW = x_sobel_CW.permute(0, 2, 1, 3)
        x_sobel_CW_a = x_sobel_CW*CW
        # x_sobel_CW_a = x_sobel_CW_a.permute(0, 2, 1, 3)
        x_sobel_HC = self.conv_HC(x.permute(0, 3, 2, 1))#Sobel_operation(x.permute(0, 3, 2, 1))  # 将特征张量转置为目标形状 (B, W, H, C)
        x_sobel_HC = x_sobel_HC.permute(0, 3, 2, 1)
        x_sobel_HC_a = x_sobel_HC*HC
        # x_sobel_HC_a = x_sobel_HC_a.permute(0, 3, 2, 1)
        xqc1 = qiancha(x,x_sobel_HW_a,x_sobel_CW_a,x_sobel_HC_a)
        xqc1 = self.ps(xqc1)
        xqc2 = qiancha(x_sobel_HC_a,x_sobel_CW_a,x_sobel_HW_a,x)
        xqc2 = self.ps(xqc2)
        xqc = qiancha2(xqc1,xqc2)
        x_sobel_a = self.conv_ps_down2(xqc)
        # x_sobel_a = self.Sobel_conv(torch.cat((x_sobel_HW_a,x_sobel_CW_a,x_sobel_HC_a),dim=1))
        out = x_sobel_a#+self.layers(x)#*x_sobel_a
        out=self.dropout(out)
        out = out + x
        out1 = self.layers(out)#*x_sobel_a
        out1=self.dropout(out1)
        out1 = x + out1
        
        return out1


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.channel=channel
        self.dropout = nn.Dropout(p=0.5)
        self.layers = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                    )
    def forward(self, x):
        out = self.layers(x)
        out=self.dropout(out)
        out = out + x
        return out

class conv3d(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.channel=channel
        self.dropout = nn.Dropout(p=0.5)
        self.layers = nn.Conv3d(self.channel, self.channel, 3, stride=1, padding=1)

    def forward(self, x):
        out = self.layers(x)
        out=self.dropout(out)
        # out = out + x
        return out

class ResCA(nn.Module):
    def __init__(self, channel):
        super(ResCA, self).__init__()
        self.channel = channel
        self.layers = nn.Sequential(ResBlock(self.channel),


                                    )
    def forward(self, x):
        out = self.layers(x)
        return out


class ResCA1(nn.Module):
    def __init__(self, channel):
        super(ResCA1, self).__init__()
        self.channel = channel
        self.layers = nn.Sequential(ResBlock(self.channel),
                                    eca_layer(48,3)
                                    )
    def forward(self, x):
        out = self.layers(x)
        return out

class ResSA(nn.Module):
    def __init__(self, channel):
        super(ResSA, self).__init__()
        self.layers = nn.Sequential(ResBlock(48)

                                    )
    def forward(self, x):
        out = self.layers(x)
        return out

class ResSA1(nn.Module):
    def __init__(self, channel):
        super(ResSA1, self).__init__()
        self.layers = nn.Sequential(ResBlock(48),
                                   SpatialGate()
                                    )
    def forward(self, x):
        out = self.layers(x)
        return out

class ResCSA(nn.Module):
    def __init__(self, channel):
        super(ResCSA, self).__init__()
        self.layers = nn.Sequential(ResBlock(48),
                                    ResBlock(48),

                                    )
    def forward(self, x):
        out = self.layers(x)
        return out

class ResCSA1(nn.Module):
    def __init__(self, channel):
        super(ResCSA1, self).__init__()
        self.layers = nn.Sequential(ResBlock(48),
                                    ResBlock(48),

                                    SpatialGate(),
                                    eca_layer(48, 3),
                                    )
    def forward(self, x):
        out = self.layers(x)
        return out

class Upsample(nn.Module):
    def __init__(self, n_feat,scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*scale*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)

class Upsample4(nn.Module):
    def __init__(self, n_feat,scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*scale*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)

class Upsamplein(nn.Module):
    def __init__(self, in_feat, out_feat,scale_f):
        super(Upsamplein, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_f, mode='bilinear', align_corners=True)
        self.body =nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.up(self.body(x))

class specconv(nn.Module):
    def __init__(self, out_feat,scale_f):
        super(specconv, self).__init__()
        min_feat=int((out_feat/scale_f)/scale_f)

        # self.up = nn.Upsample(scale_factor=scale_f, mode='bilinear', align_corners=True)
        self.body =nn.Conv2d(min_feat, out_feat, kernel_size=5, stride=1, padding=2, bias=False)

        self.ps = nn.PixelShuffle(scale_f)

        self.down1 = nn.Conv2d(out_feat, out_feat, 4, 4, 0, bias=False)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        out = self.down1(self.body(self.ps(x)))
        out=self.dropout(out)
        out = out + x
        return out

class specconv6(nn.Module):
    def __init__(self, in_feat,out_feat,scale_f):
        super(specconv6, self).__init__()
        # min_feat=int((out_feat/scale_f)/scale_f)
        self.inf =nn.Conv2d(in_feat, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # self.up = nn.Upsample(scale_factor=scale_f, mode='bilinear', align_corners=True)
        self.body =nn.Conv2d(1, out_feat, kernel_size=9, stride=1, padding=4, bias=False)

        self.ps = nn.PixelShuffle(8)

        self.down1 = nn.Conv2d(out_feat, out_feat, scale_f, scale_f, 0, bias=False)
        # self.dropout = nn.Dropout(p=0.5)
        scale_fa=int(8/scale_f)

        self.s_up = nn.Upsample(scale_factor=scale_fa, mode='bilinear', align_corners=True)
        self.s_body =nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False)

        self.f_body =nn.Conv2d(out_feat*2, out_feat, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        out1 = self.down1(self.body(self.ps(self.inf(x))))
        out2 = self.s_up(self.s_body(x))
        out = self.f_body(torch.cat((out1,out2),dim=1))
        # out=self.dropout(out)
        # out = out + x
        return out

class allnet(nn.Module):
    def __init__(self, channel):
        super(allnet, self).__init__()
        in_stage=31
        dim_stage=48
        # self.conv=nn.Conv2d(34,31,3,1,1)
        self.conv1 = nn.Conv2d(3, dim_stage, 3, 1, 1)
        # self.block2=crosslearn(dim_stage)
        
        self.up8 = specconv6(in_feat=in_stage, out_feat=dim_stage,scale_f=1)
        self.up4 = specconv6(in_feat=in_stage, out_feat=dim_stage,scale_f=2)
        self.up2 = specconv6(in_feat=in_stage, out_feat=dim_stage,scale_f=4)
        self.up0 = specconv6(in_feat=in_stage, out_feat=dim_stage,scale_f=8)
        # self.up8 = Upsamplein(in_feat=in_stage, out_feat=dim_stage,scale_f=8)
        # self.up4 = Upsamplein(in_feat=in_stage, out_feat=dim_stage,scale_f=4)
        # self.up2 = Upsamplein(in_feat=in_stage, out_feat=dim_stage,scale_f=2)
        # self.up0 = nn.Conv2d(in_stage, dim_stage, kernel_size=3, stride=1, padding=1, bias=False)

        self.convdown1 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb1 = ResBlock_changemarx(dim_stage,64)
        self.res_sconv1=specconv(dim_stage,4)
        self.down1 = nn.Conv2d(dim_stage, dim_stage, 4, 2, 1, bias=False)
        self.convdown2 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb2 = ResBlock_changemarx(dim_stage,32)
        self.res_sconv2=specconv(dim_stage,4)
        self.down2 = nn.Conv2d(dim_stage, dim_stage, 4, 2, 1, bias=False)
        self.convdown3 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb3 = ResBlock_changemarx(dim_stage,16)
        self.res_sconv3=specconv(dim_stage,4)
        self.down3 = nn.Conv2d(dim_stage, dim_stage, 4, 2, 1, bias=False)
        self.convdown4 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb4 = ResBlock_changemarx(dim_stage,8)
        self.res_sconv4=specconv(dim_stage,4)

        self.midresb1 = ResBlock_changemarx(dim_stage,64)
        self.midresb2 = ResBlock_changemarx(dim_stage,32)
        self.midresb3 = ResBlock_changemarx(dim_stage,16)
        # self.midresb4 = ResBlock(dim_stage)

        self.miupde1 = Upsample(dim_stage,2)
        self.miupde2 = Upsample(dim_stage,2)
        self.miupde3 = Upsample(dim_stage,2)

        self.midresb21 = ResBlock_changemarx(dim_stage,64)
        self.midresb22 = ResBlock_changemarx(dim_stage,32)
        # self.midresb23 = ResBlock(dim_stage)

        self.miupde21 = Upsample(dim_stage,2)
        self.miupde22 = Upsample(dim_stage,2)

        self.midresb31 = ResBlock_changemarx(dim_stage,64)

        self.miupde31 = Upsample(dim_stage,2)



        self.convout=nn.Conv2d(dim_stage, 31, 1, 1, 0, bias=False)


    def forward(self, x,y): # x:train_lrhs  y:train_hrms
        xu8=self.up8(x)
        xu4=self.up4(x)
        xu2=self.up2(x)
        xu0=self.up0(x)
        y0 =self.conv1(y)

        en0=torch.cat((xu8,y0),dim=1)
        en0=self.convdown1(en0)
        enres = en0
        en0=self.resb1(en0)
        en1=self.down1(en0)
        en1=torch.cat((xu4,en1),dim=1)
        en1=self.resb2(self.convdown2(en1))
        en2=self.down2(en1)
        en2=torch.cat((xu2,en2),dim=1)
        en2=self.resb3(self.convdown3(en2))
        en3=self.down3(en2)
        en3=torch.cat((xu0,en3),dim=1)
        en3=self.resb4(self.convdown4(en3))

        mi1=self.midresb1(en0)
        enu1=self.miupde1(en1)
        mi2=self.midresb2(en1)
        enu2=self.miupde2(en2)
        mi3=self.midresb3(en2)
        enu3=self.miupde3(en3)
        # mi4=self.midresb4(en3)

        mi21=self.midresb21(mi1+enu1)
        enu21=self.miupde21(mi2+enu2)
        mi22=self.midresb22(mi2+enu2)
        enu22=self.miupde22(mi3+enu3)
        # mi23=self.midresb23(mi3+enu3)

        mi31=self.midresb31(mi21+enu21)
        enu32=self.miupde22(mi22+enu22)


        out=self.convout(mi31+enu32)

        return out,out,out




a=torch.rand(1,31,8,8)
b=torch.rand(1,3,64,64)
cnn=allnet(48)
flops, params = profile(cnn, inputs=(a,b ))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
c,c1,c2=cnn(a,b)
print(c.shape)