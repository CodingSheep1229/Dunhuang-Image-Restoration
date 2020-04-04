import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import matplotlib.pyplot as plt
import random
import numpy as np

in_path = sys.argv[1]
out_path = sys.argv[2]

from lib.model import PConvUNet, VGG16FeatureExtractor
from utils import unnormalize
from Dataset import MyDataset

def predict(model, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    
    result = []
    for step, batch in enumerate(test_dataloader):
        img, mask = [t.type(torch.float).to(device) for t in batch]
        mask = mask.permute(0,3,1,2)
        output, _ = model(img, mask)
        output_comp = mask * img + (1 - mask) * output[0]
        result.append(output_comp.detach().cpu()[0])
        
    return result

seed = 87
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

test_dataset = MyDataset(in_path, is_train=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PConvUNet().to(device)
model.load_state_dict(torch.load('./model5.pth'))

res = predict(model, test_dataset)

for i in range(len(res)):
    file = os.path.join(out_path, test_dataset.imgs[i]+'.jpg')
    plt.imsave(file,unnormalize(res[i]))