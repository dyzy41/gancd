import os
import numpy as np
import cv2
import random
import tqdm

p = '/home/user/dsj_files/CDdata/WHUCD/image_data/cut_data'
save_dir = '/home/user/dsj_files/CDdata/WHUCD/image_data/cut_data/whub_txt'
train_info = open(os.path.join(p, 'train.txt'), 'r').readlines()
val_info = open(os.path.join(p, 'val.txt'), 'r').readlines()
test_info = open(os.path.join(p, 'test.txt'), 'r').readlines()
all_info = train_info + val_info + test_info
new_info = []

for item in tqdm.tqdm(all_info):
    lab_path = item.strip().split('  ')[2]
    label = cv2.imread(lab_path, 0)
    sum_ = np.sum(label)
    if sum_ > int(256*256*0.1):
        new_info.append(item)


random.shuffle(new_info)
new_train = new_info[:int(len(new_info)*0.7)]
new_val = new_info[int(len(new_info)*0.7):int(len(new_info)*0.8)]
new_test = new_info[int(len(new_info)*0.8):]

with open(os.path.join(save_dir, 'train.txt'), 'w') as f:
    for item in new_train:
        f.write(item)
with open(os.path.join(save_dir, 'val.txt'), 'w') as f:
    for item in new_val:
        f.write(item)
with open(os.path.join(save_dir, 'test.txt'), 'w') as f:
    for item in new_test:
        f.write(item)
