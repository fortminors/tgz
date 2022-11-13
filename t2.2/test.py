from collections import OrderedDict
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics import MeanSquaredError

from AudioDataset import AudioDataset

from SpectrogramDenoiser import SpectrogramDenoiser

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='./checkpoints/model.pt', help='Weights path')
parser.add_argument('--dataset_path', type=str, default='./train1/train', help='Dataset to run testing on')

opt = parser.parse_args()

test_dataset_path = opt.dataset_path
model_path = opt.weights

device = torch.device('cuda:0')
image_size = (1, 760, 80)

visualize = False

mse = MeanSquaredError().to(device)

model = SpectrogramDenoiser()

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

model = model.to(device)
model.eval()

test_dataset = AudioDataset(test_dataset_path, image_size)

print(f"Testing dataset size = {len(test_dataset)}")

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

true_labels = []
pred_labels = []

computed_mean = 0.436
computed_std = 0.176

with torch.no_grad():
    for i, (noisy_image, clean_image) in enumerate(tqdm(test_loader)):
        noisy_image = noisy_image.to(device)
        clean_image = clean_image.to(device)

        out = model(noisy_image)

        if (visualize):
            out_image = ((out[0] * computed_std + computed_mean) * 255.0).permute(2,1,0).detach().cpu().numpy().astype(np.uint8)
            noisy = ((noisy_image[0] * computed_std + computed_mean) * 255.0).permute(2,1,0).detach().cpu().numpy().astype(np.uint8)
            clean = ((clean_image[0] * computed_std + computed_mean) * 255.0).permute(2,1,0).detach().cpu().numpy().astype(np.uint8)

            vis = np.vstack((out_image, noisy, clean))

            cv2.imwrite(f'results/result_{i}.jpg', vis)

        mse.update(out, clean_image)

mse_result = mse.compute().item()

print(f"MSE metric = {mse_result}")
