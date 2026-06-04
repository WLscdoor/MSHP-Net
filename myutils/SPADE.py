import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, feature_size, style_size):
        super(SPADE, self).__init__()
        self.norm = nn.BatchNorm2d(feature_size, affine=False)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(style_size, 128, 3, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.conv_gamma = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))
        self.conv_beta = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))
    
    def forward(self, x, s):
        s = F.interpolate(s, size=(x.size(2), x.size(3)), mode='nearest')
        s = self.conv(s)
        return self.norm(x) * self.conv_gamma(s) + self.conv_beta(s)

class SPADEResBlk(nn.Module):
    def __init__(self, input_size, output_size, style_size):
        super(SPADEResBlk, self).__init__()
        # Main layer
        self.spade_1 = SPADE(input_size, style_size)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = spectral_norm(nn.Conv2d(input_size, output_size, 3, 1, 1))
        self.spade_2 = SPADE(output_size, style_size)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = spectral_norm(nn.Conv2d(output_size, output_size, 3, 1, 1))
        # Shortcut layer
        self.spade_s = SPADE(input_size, style_size)
        self.relu_s = nn.ReLU(inplace=True)
        self.conv_s = spectral_norm(nn.Conv2d(input_size, output_size, 3, 1, 1))
    
    def forward(self, x, s):
        y  = self.conv_1(self.relu_1(self.spade_1(x, s)))
        y  = self.conv_2(self.relu_2(self.spade_2(y, s)))
        y_ = self.conv_s(self.relu_s(self.spade_s(x, s)))
        return y + y_

# input_tensor = torch.rand(2, 16, 256, 256)
# mask_tensor = torch.rand(2, 1, 256, 256)
# module = SPADEResBlk(16, 16, 1)
# output = module(input_tensor, mask_tensor)
# print(output.size())
# aaa = 1
# aaa = SPADEResBlk(input_tensor,)