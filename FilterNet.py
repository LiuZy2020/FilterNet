import torch
import torch.nn as nn
import torch.nn.functional as torchF
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.net = nn.Sequential(  
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),## 64x64
            nn.LeakyReLU(0.2,inplace=True),
            self._block(32,  64, 3, 1, 2),#32x32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(64, 64, 3, 1, 2),#16x16
            nn.LeakyReLU(0.2, inplace=True),
            self._block(64, 128, 3, 1, 2),#8x8
            nn.LeakyReLU(0.2, inplace=True),
            self._block(128, 128, 3, 1, 2),#4x4
            nn.LeakyReLU(0.2, inplace=True),
            self._block(128, 256, 3, 1, 2),#2x2
            nn.LeakyReLU(0.2, inplace=True),
            self._block(256, self.num_classes, 3, 1, 2),#1x1
        )


    def _block(self, inchannels, outchannels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=inchannels,
                out_channels=outchannels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )


    def forward(self, image):
        image = self.net(image)
        return image

class simple_Classifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(simple_Classifier, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.Pooling = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.LinearL = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 1, out_features=8),
            nn.Linear(in_features=8, out_features=self.num_classes),
        )

    def forward(self, image):
        image = self.Pooling(image)
        image = self.LinearL(image.view(image.size(0), -1))
        return image


class encoder(nn.Module):
    def __init__(self, input_channels):
        super(encoder, self).__init__()
        self.input_channels = input_channels

        self.g_encoder = nn.Sequential(
            ## input.shape(1,128,128)
            nn.Conv2d(in_channels=self.input_channels, out_channels=16, kernel_size=3, padding=1, stride=2),
            ## 64x64
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),
            ## 32x32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            ## 16x16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            ## 8x8
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )

    def forward(self, image):
        return self.g_encoder(image)

class decoder(nn.Module):
    def __init__(self, input_channels):
        super(decoder, self).__init__()
        self.input_channels = input_channels

        self.g_decoder = nn.Sequential(
            ##3x3 output(128,16,16)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            ##5x5 output(64,32,32)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, padding=1, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            ##5x5 output(32,64,64)
            nn.ConvTranspose2d(in_channels=32, out_channels=16, padding=1, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),
            ##5x5 output(1,128,128)
            nn.ConvTranspose2d(in_channels=16, out_channels=16, padding=1, kernel_size=3, stride=2, output_padding=1),
            nn.ConvTranspose2d(in_channels=16, out_channels=self.input_channels, padding=1, kernel_size=3, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, latent):
        return self.g_decoder(latent)

def get_parameter_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_number': total_num, 'trainable_number':trainable_num}

if __name__ == '__main__':
    print(torch.__version__)
