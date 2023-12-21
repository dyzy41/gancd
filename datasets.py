import glob
import random
import os
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.infos = open(os.path.join(root, '%s.txt' % mode)).readlines()
        self.files_A = [ii.strip().split('  ')[0] for ii in self.infos]
        self.files_B = [ii.strip().split('  ')[1] for ii in self.infos]
        self.files_lab = [ii.strip().split('  ')[2] for ii in self.infos]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        lab = cv2.imread(self.files_lab[index % len(self.files_A)], 0)
        lab = lab//255

        return {'A': item_A, 'B': item_B, 'lab': lab, 'name':os.path.basename(self.files_A[index % len(self.files_A)])}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))