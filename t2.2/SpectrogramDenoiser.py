from collections import OrderedDict

import torch.nn as nn

class SpectrogramDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_downsample_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3,3), padding='same')),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(2,2))),
        ]))

        self.encoder_downsample_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding='same')),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(2,2))),
        ]))

        self.encoder_downsample_3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding='same')),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(2,2))),
        ]))

        self.decoder_upsample_1 = nn.Sequential(OrderedDict([
            ('convT2d', nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2)),
            ('relu', nn.ReLU()),
        ]))

        self.decoder_upsample_2 = nn.Sequential(OrderedDict([
            ('convT2d', nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=2)),
            ('relu', nn.ReLU()),
        ]))

        self.decoder_upsample_3 = nn.Sequential(OrderedDict([
            ('convT2d', nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2)),
            ('relu', nn.ReLU()),
        ]))

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3,3), padding='same')

    def forward(self, x):
        x = self.encoder_downsample_1(x)
        x = self.encoder_downsample_2(x)
        x = self.encoder_downsample_3(x)

        x = self.decoder_upsample_1(x)
        x = self.decoder_upsample_2(x)
        x = self.decoder_upsample_3(x)
        
        out = self.conv_out(x)

        return out
