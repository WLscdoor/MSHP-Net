import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

        # xavier_uniform_(self.attention.weight)
        # zeros_(self.attention.bias)


class deconv1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deconv1d, self).__init__()

        self.de_conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1,output_padding=1)

    def forward(self, x):
        x = self.de_conv(x)
        return x

class conv1d_block(nn.Module):
    """
    Linear Block 
    """
    def __init__(self, in_ch, out_ch, drop = 0.1, kernel_size = 3):
        super(conv1d_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=1),
            # nn.BatchNorm1d(out_ch)
            )
        for m in self.conv:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight) 
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        return x

class linear_block(nn.Module):
    """
    Linear Block 
    """
    def __init__(self, in_ch, out_ch, drop = 0.1):
        super(linear_block, self).__init__()
        c=int(out_ch/16)
        self.conv = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch)
            )

        for m in self.conv:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                # nn.init.zeros_(m.weight) 
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, kernel_size = 3):
        super(conv_block, self).__init__()
        c=int(out_ch/16)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=c),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=c))

        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) 
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_u(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, max_pool):
        super(conv_block_u, self).__init__()
        c=int(out_ch/16)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=c),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=c))
        # self.conv1 = 
        self.maxpool = nn.MaxPool2d(kernel_size=max_pool, stride=max_pool)
        self.res_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.res_skip1 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn = nn.GroupNorm(num_channels=out_ch, num_groups=int(out_ch/16))
        self.relu = nn.ReLU(inplace=True)

        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) 
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.maxpool(x)
        x1 = self.conv(x)
        # x_out = (x1+ (self.gn(self.res_skip(x))))
        x_out = x1

        return x_out

class Unet_encoder(nn.Module):
    def __init__(self, in_channel, depth):
        super(Unet_encoder, self).__init__()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        onelayer=[]
        onelayer.append(conv_block_u(in_channel, in_channel*2, 1))
        for i in range(1, depth):
            onelayer.append(conv_block_u(in_channel*(2**i), in_channel*(2**(i+1)), 2))
        self.encoder = nn.ModuleList(onelayer)
        self.depth = depth


    def forward(self,x):
        x_out = []
        x_out.append(self.encoder[0](x))
        for i in range(1, self.depth):
            x_out.append(self.encoder[i](x_out[i-1]))

        return x_out

# aa = Unet_encoder(4,16)
# input_tensor = torch.rand(2, 16, 256, 256)
# mask_tensor = torch.rand(2, 1, 256, 256)
# bb = aa(input_tensor)