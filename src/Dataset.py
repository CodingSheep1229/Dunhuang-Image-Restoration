import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, img_path, gt_path=None, is_train=True):
        imgs = [i.split('_')[0] for i in sorted(os.listdir(img_path)) if '_masked.jpg' in i]
        
        self.is_train = is_train
        self.have_gt = gt_path is not None
        self.X = []
        self.y = []
        self.mask = []
        for i in imgs:
            temp_img = Image.open(os.path.join(img_path, i+'_masked.jpg'))
            self.X.append(self.preprocess(temp_img))
            temp_img.close()
            
            temp_mask = Image.open(os.path.join(img_path, i+'_mask.jpg'))
            self.mask.append(temp_mask.copy().convert('RGB'))
            temp_mask.close()
            
            if self.have_gt:
                temp_gt = Image.open(os.path.join(gt_path, i+'.jpg'))
                self.y.append(self.preprocess(temp_gt))
                temp_gt.close()
        
        self.len = len(self.X)
        self.imgs = imgs
            
    def preprocess(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        res = trans(img)
        return np.array(res)
            
    def __getitem__(self, index):
        img = self.X[index]
        msk = (np.array(self.mask[index])>127).astype(float)
        if self.have_gt:
            y = self.y[index]
        _, h, w = img.shape
        x_start = random.randint(0,w-401)
        y_start = random.randint(0,h-401)
        
        if self.is_train:
            img = img[:,y_start:y_start+400,x_start:x_start+400]
            msk = msk[y_start:y_start+400,x_start:x_start+400,:]
            y = y[:,y_start:y_start+400,x_start:x_start+400]
        
        if self.have_gt:
            return img, msk, y
        else:
            return img, msk
    
    def __len__(self):
        return self.len