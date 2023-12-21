import os
import numpy as np
import cv2
import random
import tqdm
import shutil

p = '/home/user/dsj_files/CDdata/WHUCD/image_data/cut_data/whub_txt'
train_info = open(os.path.join(p, 'train.txt'), 'r').readlines()
val_info = open(os.path.join(p, 'val.txt'), 'r').readlines()
test_info = open(os.path.join(p, 'test.txt'), 'r').readlines()


os.makedirs(os.path.join(p, 'train/A'), exist_ok=True)
os.makedirs(os.path.join(p, 'train/B'), exist_ok=True)
os.makedirs(os.path.join(p, 'train/label'), exist_ok=True)
os.makedirs(os.path.join(p, 'val/A'), exist_ok=True)
os.makedirs(os.path.join(p, 'val/B'), exist_ok=True)
os.makedirs(os.path.join(p, 'val/label'), exist_ok=True)
os.makedirs(os.path.join(p, 'test/A'), exist_ok=True)
os.makedirs(os.path.join(p, 'test/B'), exist_ok=True)
os.makedirs(os.path.join(p, 'test/label'), exist_ok=True)

for item in tqdm.tqdm(train_info):
    pathA, pathB, pathLAB = item.strip().split('  ')
    shutil.copy(pathA, os.path.join(p, 'train/A'))
    shutil.copy(pathB, os.path.join(p, 'train/B'))
    shutil.copy(pathLAB, os.path.join(p, 'train/label'))

for item in tqdm.tqdm(val_info):
    pathA, pathB, pathLAB = item.strip().split('  ')
    shutil.copy(pathA, os.path.join(p, 'val/A'))
    shutil.copy(pathB, os.path.join(p, 'val/B'))
    shutil.copy(pathLAB, os.path.join(p, 'val/label'))

for item in tqdm.tqdm(test_info):
    pathA, pathB, pathLAB = item.strip().split('  ')
    shutil.copy(pathA, os.path.join(p, 'test/A'))
    shutil.copy(pathB, os.path.join(p, 'test/B'))
    shutil.copy(pathLAB, os.path.join(p, 'test/label'))
    
    