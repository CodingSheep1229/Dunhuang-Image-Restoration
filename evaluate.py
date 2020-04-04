import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity 

def read_image(path):
    img = plt.imread(path)
    return img 

def get_mse(img_1, img_2):
    return np.mean((img_1 - img_2) ** 2)

def get_ssim(img_1, img_2):
    return structural_similarity(img_1, img_2, multichannel= True)

def get_average_mse_ssim(fig_names_1, fig_names_2):
    mse_total = 0
    ssim_total = 0
    for i in range(len(fig_names_1)):
        img_1 = read_image(fig_names_1[i])
        img_2 = read_image(fig_names_2[i])
        mse_total  += get_mse(img_1, img_2)
        ssim_total += get_ssim(img_1, img_2)
    mse_avg  = mse_total / (i + 1)
    ssim_avg = ssim_total / (i + 1)
    return mse_avg, ssim_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", help= "Path to ground truth folder")
    parser.add_argument("-p", help= "Path to prediction folder")
    args = parser.parse_args()

    img_gt_paths = []
    img_pred_paths = []
    for i in range(100):
        img_name = "{}.jpg".format(401 + i)
        img_gt_paths.append(os.path.join(args.g, img_name))
        img_pred_paths.append(os.path.join(args.p, img_name))

    mse, ssim = get_average_mse_ssim(img_gt_paths, img_pred_paths)
    print("Average MSE: {:.5f}".format(mse))
    print("Average SSIM: {:.5f}".format(ssim))

if __name__ == "__main__":
    main()