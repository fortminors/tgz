import argparse

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.functional import precision_recall, accuracy

from AudioDataset import AudioDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./checkpoints/model.pt', help='Weights path')
    parser.add_argument('--dataset_path', type=str, default='./train1/train', help='Dataset to run testing on')

    opt = parser.parse_args()

    test_dataset_path = opt.dataset_path
    model_path = opt.weights

    # test_dataset_path = #r'C:\Users\fortm\Desktop\goznak\val\val'
    # model_path = #'checkpoints/model.pt'

    device = torch.device('cuda:0')

    model = models.resnet18()

    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

    model = model.to(device)
    model.eval()

    test_dataset = AudioDataset(test_dataset_path)

    if (len(test_dataset) > 0):
        print(f"Testing dataset size = {len(test_dataset)}")
    else:
        print(f"No samples for the dataset at the given path.")
        exit(-1)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for image, true_label in tqdm(test_loader):
            image = image.to(device)

            true_label = true_label.to(device)

            out = model(image)

            pred_label = out.argmax(axis=1)

            true_labels.append(true_label.detach().cpu())
            pred_labels.append(pred_label.detach().cpu())

    true_labels = torch.concatenate(true_labels)
    pred_labels = torch.concatenate(pred_labels)

    precision, recall = precision_recall(pred_labels, true_labels, average='macro', num_classes=2)

    acc = accuracy(pred_labels, true_labels, average='macro', num_classes=2)

    print(f"Accuracy = {acc.item()}, Precision = {precision.item()}, Recall = {recall.item()}")
