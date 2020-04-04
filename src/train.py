import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.loss import InpaintingLoss
from lib.SSIM import ssim_loss
from lib.model import PConvUNet, VGG16FeatureExtractor
from Dataset import MyDataset
from utils import score
import random
import numpy as np

seed = 87
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class Args:
    pass
args = Args()
args.lr = 4e-4
args.save_path = 'model.pth'
args.batch_size = 8
args.epochs = 200
args.LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
args.accul_grad = 1

def evaluate(model, test_dataset, args, epoch):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)
    model.eval()
    val_loss = 0.
    val_mse_score = 0.
    val_ssim_score = 0.
    
    for step, batch in enumerate(test_dataloader):
        img, mask, gt = [t.type(torch.float).to(device) for t in batch]
        mask = mask.permute(0,3,1,2)
        output, _ = model(img, mask)
        loss_dict = criterion(img, mask, output, gt)
        loss = 0.0
        for key, coef in args.LAMBDA_DICT.items():
            loss += coef * loss_dict[key].item()
            
        mse, ssim = score(output.detach().cpu().numpy(), gt.detach().cpu().numpy())
        val_mse_score += mse
        val_ssim_score += ssim
        val_loss += loss
        del loss_dict
        del loss
        torch.cuda.empty_cache()
        
    return val_loss, val_mse_score, val_ssim_score

def fit(model, train_dataset, test_dataset, args):
    his = (
        {'train_loss': [],
        'train_mse_score': [],
        'train_ssim_score': [],
        'val_loss': [],
        'val_mse_score': [],
        'val_ssim_score': [],
    })
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)
    accul_cnt = 0
    max_score = 0
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.
        tr_mse_score = 0.
        tr_ssim_score = 0.
        
        loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            img, mask, gt = [t.type(torch.float).to(device) for t in batch]
            mask = mask.permute(0,3,1,2)
            output, _ = model(img, mask)
            
            loss_dict = criterion(img, mask, output, gt)

            for key, coef in args.LAMBDA_DICT.items():
                loss += coef * loss_dict[key] / args.accul_grad
                
            loss -= 5*ssim_loss(output, gt)
            
            loss.backward()
            
            tr_loss += loss.item()
            del loss
            loss = 0.0
            
            del loss_dict
            if accul_cnt == args.accul_grad:
                accul_cnt = 0
                optimizer.step()
                optimizer.zero_grad()
        
            torch.cuda.empty_cache()    
            accul_cnt += 1
            
            mse, ssim = score(output.detach().cpu().numpy(), gt.detach().cpu().numpy())
            tr_mse_score += mse
            tr_ssim_score += ssim

        val_loss, val_mse_score, val_ssim_score = evaluate(model, test_dataset, args, epoch)
        
        tr_mse_score /= len(train_dataset)
        tr_ssim_score /= len(train_dataset)
        tr_loss /= len(train_dataset)

        val_mse_score /= len(test_dataset)
        val_ssim_score /= len(test_dataset)
        val_loss /= len(test_dataset)
        
        sc = 1 - val_mse_score/100 + val_ssim_score
        print(f'[Epoch {epoch+1}] loss: {tr_loss:.3f}, mse_score: {tr_mse_score:.3f}, ssim_score: {tr_ssim_score:.3f}, '+
              f'val_loss: {val_loss:.3f}, val_mse_score: {val_mse_score:.3f}, val_ssim_score: {val_ssim_score:.3f}', flush=True)

        his['train_loss'].append(tr_loss)
        his['train_mse_score'].append(tr_mse_score)
        his['train_ssim_score'].append(tr_ssim_score)
        his['val_loss'].append(val_loss)
        his['val_mse_score'].append(val_mse_score)
        his['val_ssim_score'].append(val_ssim_score)
        
        if sc > max_score:
            print('model saved!')
            max_score = sc
            torch.save(model.state_dict(), args.save_path)
        
    return his

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = MyDataset('../../Data_Challenge2/train','../../Data_Challenge2/train_gt',is_train=True)
test_dataset = MyDataset('../../Data_Challenge2/test','../../Data_Challenge2/test_gt', is_train=False)

model = PConvUNet().to(device)

his = fit(model, train_dataset, test_dataset, args)