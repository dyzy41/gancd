#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import wandb
import torch
import os

from models import Generator
from models import Discriminator
from cdnet import CDNet
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from PIL import Image
import numpy as np
from torchmetrics import ConfusionMatrix
from metric import CM2Metric


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/user/dsj_files/CDdata/WHUCD/image_data/cut_data/whub_txt', help='root directory of the dataset')
parser.add_argument('--save_dir', type=str, default='./output_cd', help='save directions')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)



###### Definition of variables ######
# Networks
netCD = CDNet()

if opt.cuda:
    netCD.cuda()


# Lossess
criterion_CD = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_CD = torch.optim.Adam(netCD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
lr_scheduler_CD = torch.optim.lr_scheduler.LambdaLR(optimizer_CD, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

# Dataset loader
transforms_ = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
val_dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True, mode='val'), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    netCD.train()
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()
        label = batch['lab'].cuda().long()

        optimizer_CD.zero_grad()
        predA = netCD(real_A, real_B)
        loss_CD = criterion_CD(predA, label)
        loss_CD.backward()
        optimizer_CD.step()

        logger.log({'loss_CD': loss_CD}, 
                    images={'real_A': real_A, 'real_B': real_B})
    # Update learning rates
    lr_scheduler_CD.step()

    print('eval step **************************************************')
    netCD.eval()

    os.makedirs(os.path.join(opt.save_dir, str(epoch)), exist_ok=True)
    cmA = torch.zeros((2, 2))
    metric_func = ConfusionMatrix(task="binary", num_classes=2)
    for i, batch in enumerate(val_dataloader):
        # Set model input
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()
        label = batch['lab'].long()

        predA = netCD(real_A, real_B)
        predA = torch.argmax(predA, dim=1)
        cmA+=metric_func(predA.detach().cpu(), label)

    metrics = CM2Metric(cmA.cpu().numpy())
    print('#######################################################')
    names = 'A_Pred--->OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
    for i in range(len(names)):
        print(names[i]+': '+str(metrics[i]))
    print('#######################################################')

    # Save models checkpoints
    torch.save(netCD.state_dict(), os.path.join(opt.save_dir, 'netCD.pth'))
###################################
