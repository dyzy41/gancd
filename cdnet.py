import torch.nn as nn
import torch.nn.functional as F
import torch
import timm
from timm.models._efficientnet_blocks import InvertedResidual


class BaseConv(nn.Module):
    """
    Base convolution module.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, optional): Zero-padding added to both sides of the input.
            Default: 0
        dilation (int, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels
            to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              groups,
                              bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelExchange(nn.Module):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        
        return out_x1, out_x2


class SpatialExchange(nn.Module):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        
        return out_x1, out_x2


class CDNet(nn.Module):
    def __init__(self):
        super(CDNet, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)

        # FPN_DICT = {'type': 'FPN', 'in_channels': [16, 24, 40, 80, 112, 192, 320], 'out_channels': 128, 'num_outs': 7}
    

        self.conv5 = InvertedResidual(640, 112*2)
        self.conv4 = InvertedResidual(112*2*2, 40*2)
        self.conv3 = InvertedResidual(40*2*2, 24*2)
        self.conv2 = InvertedResidual(24*2*2, 16*2)
        self.conv1 = InvertedResidual(16*2*2, 2)
    

    def forward(self, xA, xB):
        xA_list = self.model(xA)
        xB_list = self.model(xB)
        c1, c2, c3, c4, c5 = [torch.cat([xA_list[i], xB_list[i]], dim=1) for i in range(len(xA_list))]

        d4 = self.conv5(c5)
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        c4 = torch.cat([c4, d4], dim=1)
        d3 = self.conv4(c4)
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        c3 = torch.cat([c3, d3], dim=1)
        d2 = self.conv3(c3)
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        c2 = torch.cat([c2, d2], dim=1)
        d1 = self.conv2(c2)
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        c1 = torch.cat([c1, d1], dim=1)
        d0 = self.conv1(c1)
        out = F.interpolate(d0, scale_factor=2, mode='bilinear', align_corners=True)

        return out


if __name__ == '__main__':
    xA = torch.rand(2, 3, 256, 256)
    xB = torch.rand(2, 3, 256, 256)
    model = CDNet()
    out = model(xA, xB)
    print(out.shape)