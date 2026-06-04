
import datetime
import os
import time
import glob
import yaml
from tensorboardX import SummaryWriter
from datetime import datetime
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class conv_block_u(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block_u, self).__init__()
        c=int(out_ch/16)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.GroupNorm(num_channels=out_ch, num_groups= c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.GroupNorm(num_channels=out_ch, num_groups=c))
            #nn.ReLU(inplace=True))

        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) 
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.GroupNorm(num_channels=out_ch, num_groups=out_ch/16),
            nn.ReLU(inplace=True)
        )

        for layer in self.up:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight) 
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0.0)
    def forward(self, x):
        x = self.up(x)
        return x

class deconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deconv, self).__init__()

        self.de_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1,output_padding=1)

    def forward(self, x):
        x = self.de_conv(x)
        return x


        
class encoder(nn.Module):
    def __init__(self, in_channels=2, filters=[]):
        super(encoder, self).__init__()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_u(in_channels, filters[0])
        self.Conv2 = conv_block_u(filters[0], filters[1])
        self.Conv3 = conv_block_u(filters[1], filters[2])

        self.res_skip_1 = nn.Conv2d(in_channels, filters[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.gn_1 = nn.GroupNorm(num_channels=filters[0], num_groups= int(filters[0]/16))
        self.relu_1 = nn.ReLU(inplace=True)
        self.res_skip_2 = nn.Conv2d(filters[0], filters[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.gn_2 = nn.GroupNorm(num_channels=filters[1], num_groups= int(filters[1]/16))
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.Conv1(x)
        res1 = self.relu_1(e1+ self.gn_1(self.res_skip_1(x)))

        #e2 = self.Maxpool1(e1)
        e2_d = self.Maxpool1(res1)
        e2 = self.Conv2(e2_d)
        res2 = self.relu_2(e2+ self.gn_2(self.res_skip_2(e2_d)))

        e3 = self.Maxpool2(res2)
        #e3 = self.Conv3(e3)

        return [e1, e2, e3]

class decoder(nn.Module):
    def __init__(self, out_channels=2, filters=[]):
        super(decoder, self).__init__()

        self.conv3 = conv_block_u(filters[1], filters[2])
        self.Up3 = deconv(filters[2], filters[1]) # up+conv
        self.Up_conv3 = conv_block_u(filters[2], filters[1])

        self.Up2 = deconv(filters[1], filters[0])
        self.Up_conv2 = conv_block_u(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        
        self.res_skip_3 = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.gn_3 = nn.GroupNorm(num_channels=filters[2], num_groups= int(filters[2]/16))
        self.relu_3 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_1 = nn.ReLU(inplace=True)

    def forward(self, en_feat):
        e3=en_feat[2]
        e2=en_feat[1]
        e1=en_feat[0]

        d3 = self.relu_3(self.conv3(e3)+ (self.gn_3(self.res_skip_3(e3))))
        d3 = self.Up3(d3)

        d3_c = torch.cat((e2, d3), dim=1)
        d3_res = self.relu_2(self.Up_conv3(d3_c)+d3)
        d2 = self.Up2(d3_res)
        
        d2_c = torch.cat((e1, d2), dim=1)
        d2_res = self.relu_1(self.Up_conv2(d2_c))

        out = self.Conv(d2_res)

        return out

class network(nn.Module):
    def __init__(self, in_channels=2, nd=5, n_class = 4):
        super(network, self).__init__()
        filters=[32,64,128]
        self.nd = nd
        encoders = []
        rec_decoders = []
        seg_decoders = []
        dcs = []
        for i in range(nd):
            encoders.append(encoder(2, filters))
            rec_decoders.append(decoder(2, filters))
            seg_decoders.append(decoder(n_class, filters))
            dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.encoders = nn.ModuleList(encoders)
        self.rec_decoders = nn.ModuleList(rec_decoders)
        self.seg_decoders = nn.ModuleList(seg_decoders)
        self.sigmoid = nn.Sigmoid()
        self.dcs = dcs
        self.seg_conv = nn.Conv2d(self.nd * n_class, n_class, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x,k, m, label):
        en_0 = self.encoders[0](x)
        rec_0 = self.rec_decoders[0](en_0)
        seg_0 = self.seg_decoders[0](en_0)
        rec_0 = x+rec_0
        out_0 = self.dcs[0].perform(rec_0, k, m)

        en_1 = self.encoders[1](out_0)
        rec_1 = self.rec_decoders[1](en_1)
        seg_1 = self.seg_decoders[1](en_1)
        rec_1 = x+rec_1
        out_1 = self.dcs[1].perform(rec_1, k, m)       

        en_2 = self.encoders[2](out_1)
        rec_2 = self.rec_decoders[2](en_2)
        seg_2 = self.seg_decoders[2](en_2)
        rec_2 = x+rec_2
        out_2 = self.dcs[2].perform(rec_2, k, m)               
 
        en_3 = self.encoders[3](out_2)
        rec_3 = self.rec_decoders[3](en_3)
        seg_3 = self.seg_decoders[3](en_3)
        rec_3 = x+rec_3
        out_3 = self.dcs[3].perform(rec_3, k, m)          

        en_4 = self.encoders[4](out_3)
        rec_4 = self.rec_decoders[4](en_4)
        seg_4 = self.seg_decoders[4](en_4)
        rec_4 = x+rec_4
        out_4 = self.dcs[4].perform(rec_4, k, m)    

        seg_input = torch.cat((seg_0,seg_1,seg_2,seg_3,seg_4),dim=1)
        seg_out = self.seg_conv(seg_input)

        return out_4,[seg_0, seg_1, seg_2, seg_3,seg_4,seg_out]


class rec_network(nn.Module):
    def __init__(self, in_channels=2, nd=5, n_class = 4):
        super(rec_network, self).__init__()
        filters=[32,64,128]
        self.nd = nd
        encoders = []
        rec_decoders = []
        dcs = []
        for i in range(nd):
            encoders.append(encoder(2, filters))
            rec_decoders.append(decoder(2, filters))
            dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.encoders = nn.ModuleList(encoders)
        self.rec_decoders = nn.ModuleList(rec_decoders)
        self.dcs = dcs

    def forward(self, x,k, m, label):
        en_0 = self.encoders[0](x)
        rec_0 = self.rec_decoders[0](en_0)
        rec_0 = x+rec_0
        out_0 = self.dcs[0].perform(rec_0, k, m)

        en_1 = self.encoders[1](out_0)
        rec_1 = self.rec_decoders[1](en_1)
        rec_1 = x+rec_1
        out_1 = self.dcs[1].perform(rec_1, k, m)       

        en_2 = self.encoders[2](out_1)
        rec_2 = self.rec_decoders[2](en_2)
        rec_2 = x+rec_2
        out_2 = self.dcs[2].perform(rec_2, k, m)               
 
        en_3 = self.encoders[3](out_2)
        rec_3 = self.rec_decoders[3](en_3)
        rec_3 = x+rec_3
        out_3 = self.dcs[3].perform(rec_3, k, m)          

        en_4 = self.encoders[4](out_3)
        rec_4 = self.rec_decoders[4](en_4)
        rec_4 = x+rec_4
        out_4 = self.dcs[4].perform(rec_4, k, m)    

        return out_4,[label,label,label]


class  simple_U(nn.Module):
    def __init__(self, in_channels=2, nd=5, n_class = 4):
        super(simple_U, self).__init__()
        filters=[32,64,128]
        encoders = []
        seg_decoders = []
        encoders.append(encoder(in_channels, filters))
        seg_decoders.append(decoder(n_class, filters))

        self.encoders = nn.ModuleList(encoders)
        self.seg_decoders = nn.ModuleList(seg_decoders)

    def forward(self, x):
        en_0 = self.encoders[0](x)
        seg_0 = self.seg_decoders[0](en_0)

        return seg_0
    
input_tensor = torch.rand(2, 1, 256, 256)
mask_tensor = torch.rand(2, 1, 256, 256)
module = simple_U(in_channels=1, n_class=1)
output = module(input_tensor)
# print(output.size())
# aaa = 1
# aaa = SPADEResBlk(input_tensor,)

