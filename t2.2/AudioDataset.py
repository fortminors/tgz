from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class AudioDataset(Dataset):
    def __init__(self, dataset_path, image_size=None):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size

        self.files = list(self.dataset_path.joinpath('noisy').rglob('*.npy'))

        self.computed_mean = 0.436
        self.computed_std = 0.176

        self.transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[self.computed_mean], std=[self.computed_std])
                                            ])

    def __len__(self):
        return len(self.files)

    def SpectogramToImage(self, spec):
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec - spec.min()
        spec = (spec / (spec.max() + 1e-6))# * 255
        #spec = spec.astype(np.uint8)
        return spec

    def PadOrCropImage(self, image: torch.tensor):
        '''
        Pads or crops the image to the required size (padding with zeros)
        '''

        if (self.image_size is None):
            return image

        image_height = image.shape[1]
        image_width = image.shape[2]

        required_height = self.image_size[1]
        required_width = self.image_size[2]

        out = torch.zeros(self.image_size, device=image.device, dtype=image.dtype)

        max_height = min(image_height, required_height)
        max_width = min(image_width, required_width)

        out[:, :max_height, :max_width] = image[:, :max_height, :max_width]

        return out

    def LoadImage(self, path):
        spectogram = np.load(path)
        image = self.SpectogramToImage(spectogram)
        image = self.transform(image)
        image = self.PadOrCropImage(image)

        return image.float()

    def __getitem__(self, idx):
        noisy_file_path = self.files[idx]

        noisy_image = self.LoadImage(noisy_file_path)

        clean_file_path = Path(*noisy_file_path.parts[:-3]).joinpath('clean').joinpath(*noisy_file_path.parts[-2:])

        clean_image = self.LoadImage(clean_file_path)

        return noisy_image, clean_image
