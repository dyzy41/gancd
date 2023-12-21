import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet import *
from mmseg.registry import MODELS
from timm.models._efficientnet_blocks import InvertedResidual


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        self.model1_1 = nn.ReflectionPad2d(3)
        self.model1_2 = nn.Conv2d(input_nc, 64, 7)
        self.model1_3 = nn.InstanceNorm2d(64)
        self.model1_4 = nn.ReLU(inplace=True)

        # Downsampling
        in_features = 64
        out_features = in_features*2
        self.model2 = nn.ModuleList()
        for _ in range(2):
            self.model2.append(
                        nn.Sequential(
                            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(inplace=True))
                        )
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        self.model3 = nn.ModuleList()
        for _ in range(n_residual_blocks):
            self.model3.append(ResidualBlock(in_features))

        # Upsampling
        out_features = in_features//2
        self.model4 = nn.ModuleList()
        for _ in range(2):
            self.model4.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(inplace=True))
                        )
            in_features = out_features
            out_features = in_features//2

        # Output layer
        self.model5 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
            )

    def forward(self, x):
        encode_list = []
        x = self.model1_1(x)
        x = self.model1_2(x)
        x = self.model1_3(x)
        x = self.model1_4(x)
        for layer in self.model2:
            x = layer(x)
        for i, layer in enumerate(self.model3):
            if i>4:
                x = layer(x)
                encode_list.append(x)
            else:
                x = layer(x)
        down_x = x
        for layer in self.model4:
            x = layer(x)
        x = self.model5(x)
        return x, down_x


class CDNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = ResNet18()
        FPN_DICT = {'type': 'FPN', 'in_channels': [64, 128, 256, 512], 'out_channels': 128, 'num_outs': 4}
        self.fpnA = MODELS.build(FPN_DICT)
        self.fpnB = MODELS.build(FPN_DICT)
        self.decode_layers = nn.Sequential(
            InvertedResidual(256, 256),
            InvertedResidual(256, 256),
            InvertedResidual(256, 256),
            InvertedResidual(256, 256)
        )
        self.out_conv = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0)


    def forward(self, xA, xB):
        xA_list = self.backbone(xA)
        xB_list = self.backbone(xB)
        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnB(xB_list)

        curAB3 = torch.cat([xA_list[3], xB_list[3]], dim=1)
        curAB3 = self.decode_layers[3](curAB3)
        curAB3 = F.interpolate(curAB3, scale_factor=2, mode='bilinear', align_corners=False)

        curAB2 = torch.cat([xA_list[2], xB_list[2]], dim=1)
        curAB2 = curAB3+self.decode_layers[2](curAB2)
        curAB2 = F.interpolate(curAB2, scale_factor=2, mode='bilinear', align_corners=False)

        curAB1 = torch.cat([xA_list[1], xB_list[1]], dim=1)
        curAB1 = curAB2+self.decode_layers[1](curAB1)
        curAB1 = F.interpolate(curAB1, scale_factor=2, mode='bilinear', align_corners=
                                False)
        
        curAB0 = torch.cat([xA_list[0], xB_list[0]], dim=1)
        curAB0 = curAB1+self.decode_layers[0](curAB0)
        curAB0 = F.interpolate(curAB0, scale_factor=2, mode='bilinear', align_corners=
                                False)
        
        out = self.out_conv(curAB0)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

class CycleGANCD(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.netG_A2B = Generator(input_nc, output_nc)
        self.netG_B2A = Generator(output_nc, input_nc)
        self.netD_A = Discriminator(input_nc)
        self.netD_B = Discriminator(output_nc)
        self.netCD = CDNet()

    def forward_GAN(self, real_A, real_B):
        fake_B, down_A = self.netG_A2B(real_A)
        pred_fakeA2B = self.netD_B(fake_B)

        fake_A, down_B = self.netG_B2A(real_B)
        pred_fakeB2A = self.netD_A(fake_A)

        return fake_A, fake_B, down_A, down_B, pred_fakeA2B, pred_fakeB2A

    def forward_Identity(self, real_A, real_B):
        same_B, _ = self.netG_A2B(real_B)
        same_A, _ = self.netG_B2A(real_A)
        return same_A, same_B

    def forward_Cycle(self, fake_A, fake_B):
        recovered_A, _ = self.netG_B2A(fake_B)
        recovered_B, _ = self.netG_A2B(fake_A)
        return recovered_A, recovered_B

    def forward_DiscriminatorA(self, real_A, fake_A):
        pred_realA = self.netD_A(real_A)
        pred_fakeA = self.netD_A(fake_A.detach())
        return pred_realA, pred_fakeA
    
    def forward_DiscriminatorB(self, real_B, fake_B):
        pred_realB = self.netD_B(real_B)
        pred_fakeB = self.netD_B(fake_B.detach())
        return pred_realB, pred_fakeB

    def forward_CD(self, encode_list_A, encode_list_B):
        out = self.netCD(encode_list_A, encode_list_B)
        return out


if __name__ == '__main__':
    x = torch.rand(2, 3, 256, 256)
    model = Generator(3, 3)
    y = model(x)
    print(y.shape)
