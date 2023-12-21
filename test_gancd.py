#!/usr/bin/python3

import argparse

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import torch
import os
import tqdm
import cv2

from models import Generator
from cdnet import CDNet
from utils import Logger
from datasets import ImageDataset
from PIL import Image
import numpy as np
from torchmetrics import ConfusionMatrix
from metric import CM2Metric
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/user/dsj_files/CDdata/WHUCD/image_data/cut_data/whub_txt', help='root directory of the dataset')
parser.add_argument('--save_dir', type=str, default='./output', help='save directions')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

def wandb_show_img(real, fake, recover, caption=''):

    real = real[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    fake = fake[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    recover = recover[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    ori_bgc = np.concatenate([real, fake, recover], axis=1)
    ori_bgc = Image.fromarray(ori_bgc.astype('uint8'))
    wandb.Image(ori_bgc, caption=caption)
    wandb.log({"examples": [wandb.Image(ori_bgc, caption=caption)]})

def save_gan_pred(fakeA, fakeB, name):

    fakeA = 0.5*fakeA.data+0.5
    fakeB = 0.5*fakeB.data+0.5
    for i in range(len(name)):
        save_image(fakeA[i], os.path.join(opt.save_dir, 'test_pred/fakeA', name[i]))
        save_image(fakeB[i], os.path.join(opt.save_dir, 'test_pred/fakeB', name[i]))

def save_pred(predA, predB, name):
    predA = predA.detach().cpu().numpy()
    predB = predB.detach().cpu().numpy()
    for i in range(len(name)):
        cv2.imwrite(os.path.join(opt.save_dir, 'test_pred/predA', name[i]), (predA[i]*255).astype(np.uint8))
        cv2.imwrite(os.path.join(opt.save_dir, 'test_pred/predB', name[i]), (predB[i]*255).astype(np.uint8))

# wandb.init(project="CycleGAN Project", name='cycleganCD', config=opt)


###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netCD = CDNet()

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netCD.cuda()

# Dataset loader
transforms_ = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
test_dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True, mode='test'), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(test_dataloader))
###################################

###### Training ######
netG_A2B.eval()
netG_B2A.eval()
netCD.eval()

netG_A2B_state_dict = torch.load(os.path.join(opt.save_dir, 'netG_A2B.pth'))
netG_B2A_state_dict = torch.load(os.path.join(opt.save_dir, 'netG_B2A.pth'))
netCD_state_dict = torch.load(os.path.join(opt.save_dir, 'netCD.pth'))

netG_A2B.load_state_dict(netG_A2B_state_dict)
netG_B2A.load_state_dict(netG_B2A_state_dict)
netCD.load_state_dict(netCD_state_dict)

os.makedirs(os.path.join(opt.save_dir, 'test_pred'), exist_ok=True)
os.makedirs(os.path.join(opt.save_dir, 'test_pred/predA'), exist_ok=True)
os.makedirs(os.path.join(opt.save_dir, 'test_pred/predB'), exist_ok=True)
os.makedirs(os.path.join(opt.save_dir, 'test_pred/fakeA'), exist_ok=True)
os.makedirs(os.path.join(opt.save_dir, 'test_pred/fakeB'), exist_ok=True)

cmA = torch.zeros((2, 2))
cmB = torch.zeros((2, 2))
metric_func = ConfusionMatrix(task="binary", num_classes=2)

for i, batch in tqdm.tqdm(enumerate(test_dataloader)):
    # Set model input
    real_A = batch['A'].cuda()
    real_B = batch['B'].cuda()
    label = batch['lab'].long()

    fake_B = netG_A2B(real_A)
    fake_A = netG_B2A(real_B)

    predA = netCD(real_A, fake_A.detach())
    predB = netCD(fake_B.detach(), real_B)
    predA = torch.argmax(predA, dim=1)
    predB = torch.argmax(predB, dim=1)
    save_pred(predA, predB, batch['name'])
    save_gan_pred(fake_A, fake_B, batch['name'])

    cmA+=metric_func(predA.detach().cpu(), label)
    cmB+=metric_func(predB.detach().cpu(), label)

metrics = CM2Metric(cmA.cpu().numpy())
print('#######################################################')
names = 'A_Pred--->OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
for i in range(len(names)):
    print(names[i]+': '+str(metrics[i]))
metrics = CM2Metric(cmB.cpu().numpy())
names = 'B_Pred--->OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
for i in range(len(names)):
    print(names[i]+': '+str(metrics[i]))
print('#######################################################')


