#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import wandb
import torch
import os

from model_gan import CycleGANCD
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
parser.add_argument('--save_dir', type=str, default='./output_gancd_v2', help='save directions')
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


# wandb.init(project="CycleGAN Project", name='cycleganCD', config=opt)


###### Definition of variables ######
# Networks
model = CycleGANCD(opt.input_nc, opt.output_nc)

if opt.cuda:
    model.cuda()


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_CD = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(model.netG_A2B.parameters(), model.netG_A2B.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(model.netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(model.netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_CD = torch.optim.Adam(model.netCD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
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
    model.train()
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()
        label = batch['lab'].cuda().long()

        optimizer_G.zero_grad()

        # Identity loss
        same_A, same_B = model.forward_Identity(real_A, real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_A, fake_B, down_A, down_B, pred_fakeA2B, pred_fakeB2A = model.forward_GAN(real_A, real_B)
        loss_GAN_A2B = criterion_GAN(pred_fakeA2B, target_real)
        loss_GAN_B2A = criterion_GAN(pred_fakeB2A, target_real)

        # Cycle loss
        recovered_A, recovered_B = model.forward_Cycle(fake_A, fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()

        ###################################
        # Change Detection loss

        optimizer_CD.zero_grad()

        pred = model.forward_CD(down_A.detach(), down_B.detach())

        loss_CD = criterion_CD(pred, label)
        loss_CD.backward()
        optimizer_CD.step()

        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()
        pred_realA, pred_fakeA = model.forward_DiscriminatorA(real_A, fake_A)
        # Real loss
        loss_D_real = criterion_GAN(pred_realA, target_real)
        # Fake loss
        loss_D_fake = criterion_GAN(pred_fakeA, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()
        pred_realB, pred_fakeB = model.forward_DiscriminatorB(real_B, fake_B)
        # Real loss
        loss_D_real = criterion_GAN(pred_realB, target_real)        
        # Fake loss
        loss_D_fake = criterion_GAN(pred_fakeB, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B), 
                    'loss_CD': loss_CD}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        # break
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_CD.step()

    print('eval step **************************************************')
    model.eval()

    os.makedirs(os.path.join(opt.save_dir, str(epoch)), exist_ok=True)
    cm0 = torch.zeros((2, 2))

    metric_func = ConfusionMatrix(task="binary", num_classes=2)
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            # Set model input
            real_A = batch['A'].cuda()
            real_B = batch['B'].cuda()
            label = batch['lab'].long()

            fake_A, fake_B, encode_list_A, encode_list_B, pred_fakeA2B, pred_fakeB2A = model.forward_GAN(real_A, real_B)

            pred = model.forward_CD(encode_list_A, encode_list_B)
            pred = torch.argmax(pred, dim=1)

            cm0+=metric_func(pred.detach().cpu(), label)

    print('#######################################################')
    metrics = CM2Metric(cm0.cpu().numpy())
    names = 'A_Pred--->OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
    for i in range(len(names)):
        print(names[i]+': '+str(metrics[i]))
    print('#######################################################')
    # Save models checkpoints
    torch.save(model.state_dict(), os.path.join(opt.save_dir, 'model.pth'))
###################################
