import torch
import numpy as np
from skimage.metrics import structural_similarity

def unnormalize(x):
    x = torch.FloatTensor(x)
    x = x.permute(1,2,0)
    x = x * torch.FloatTensor([0.229, 0.224, 0.225]) + torch.FloatTensor([0.485, 0.456, 0.406])
    x *= 255
    return x.numpy().astype(np.uint8)

def score(pred, gt):
    def get_mse(pair):
        img_1, img_2 = pair[0],pair[1]
        return np.mean((img_1 - img_2) ** 2)

    def get_ssim(pair):
        img_1, img_2 = pair[0],pair[1]
        return structural_similarity(img_1, img_2, multichannel= True)
    
    mses = 0
    ssims = 0
    for i in range(len(pred)):
        pair = (unnormalize(pred[i]), unnormalize(gt[i]))
        mses += get_mse(pair)
        ssims += get_ssim(pair)
 
    return mses, ssims

