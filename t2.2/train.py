from collections import OrderedDict
import argparse

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchinfo import summary

import wandb

from AudioDataset import AudioDataset
from SpectrogramDenoiser import SpectrogramDenoiser

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', type=str, default='./train1/train', help='Dataset to run training on')
    parser.add_argument('--val_dataset_path', type=str, default='./val/val', help='Dataset to run validation on')

    opt = parser.parse_args()

    train_dataset_path = opt.train_dataset_path
    val_dataset_path = opt.val_dataset_path

    lr = 3e-4
    epochs = 10
    batch_size = 32
    image_size = (1, 760, 80)

    device = torch.device('cuda:0')

    model = SpectrogramDenoiser()

    model = model.to(device)

    train_dataset = AudioDataset(train_dataset_path, image_size)
    val_dataset = AudioDataset(val_dataset_path, image_size)

    if (len(train_dataset) > 0 and len(val_dataset) > 0):
        print(f"Train dataset size = {len(train_dataset)}, Val dataset size = {len(val_dataset)}")
    else:
        print(f"No samples for the datasets at the given paths.")
        exit(-1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)
    optimizer.zero_grad()
    
    # Every 2 epochs
    lr_step_size = 2 * epochs * len(train_dataset) 

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    wandb.config = {
        "model": 'denoise',
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "optimizer": str(optimizer),
        "lr_scheduler": str(lr_scheduler),
        "criterion": str(criterion),
    }

    wandb.init(project="noise-detector", entity="georgygunkin", config=wandb.config)

    for epoch in range(epochs):
        print(f"Epoch {epoch}:")

        train_losses = []

        model.train()

        for noisy_image, clean_image in tqdm(train_loader):
            noisy_image = noisy_image.to(device)
            clean_image = clean_image.to(device)

            out = model(noisy_image)

            loss = criterion(out, clean_image)

            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_train_loss = sum(train_losses) / len(train_losses)

        print(f"Mean train loss = {mean_train_loss}")

        wandb.log({"train_loss": mean_train_loss})

        val_losses = []

        model.eval()

        with torch.no_grad():
            for noisy_image, clean_image in tqdm(val_loader):
                noisy_image = noisy_image.to(device)
                clean_image = clean_image.to(device)

                out = model(noisy_image)

                loss = criterion(out, clean_image)

                val_losses.append(loss.item())

        mean_val_loss = sum(val_losses) / len(val_losses)

        wandb.log({"val_loss": mean_val_loss})

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f"checkpoints/model_{epoch}.pt")
