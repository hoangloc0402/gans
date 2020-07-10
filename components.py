# REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def get_data_loader(train, batch_size=128):
    root = '/home/loc/data'
    dataset = MNIST(root,
                    train=train,
                    transform=transforms.Compose([transforms.Resize(64),
                                                  transforms.ToTensor(),
                                                 ]),
                    target_transform=None,
                    download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


def set_random_seed(value):
    random.seed(value)
    torch.manual_seed(value)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    INPUT_SIZE = 100
    FEATURE_MAP_SIZE = 64
    NUM_OUTPUT_CHANNEL = 1

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels = self.INPUT_SIZE,
                               out_channels = self.FEATURE_MAP_SIZE * 8,
                               kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(self.FEATURE_MAP_SIZE * 8),
            nn.ReLU(True),
            # (FEATURE_MAP_SIZE*8) x 4 x 4
            nn.ConvTranspose2d(in_channels = self.FEATURE_MAP_SIZE * 8,
                               out_channels = self.FEATURE_MAP_SIZE * 4,
                               kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(self.FEATURE_MAP_SIZE * 4),
            nn.ReLU(True),
            # (FEATURE_MAP_SIZE*4) x 8 x 8
            nn.ConvTranspose2d(in_channels = self.FEATURE_MAP_SIZE * 4,
                               out_channels = self.FEATURE_MAP_SIZE * 2,
                               kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(self.FEATURE_MAP_SIZE * 2),
            nn.ReLU(True),
            # (FEATURE_MAP_SIZE*2) x 16 x 16
            nn.ConvTranspose2d(in_channels = self.FEATURE_MAP_SIZE * 2,
                               out_channels = self.FEATURE_MAP_SIZE,
                               kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(self.FEATURE_MAP_SIZE),
            nn.ReLU(True),
            # (FEATURE_MAP_SIZE) x 32 x 32
            nn.ConvTranspose2d(in_channels = self.FEATURE_MAP_SIZE,
                               out_channels = self.NUM_OUTPUT_CHANNEL,
                               kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.Tanh()
            # (NUM_OUTPUT_CHANNEL) x 64 x 64
        )
        self.init_weights()

    def forward(self, x):
        return self.main(x)

    def init_weights(self):
        self.apply(weights_init)


class Discriminator(nn.Module):
    FEATURE_MAP_SIZE = Generator.FEATURE_MAP_SIZE
    NUM_INPUT_CHANNEL = Generator.NUM_OUTPUT_CHANNEL

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # (NUM_INPUT_CHANNEL) x 64 x 64
            nn.Conv2d(in_channels = self.NUM_INPUT_CHANNEL,
                      out_channels = self.FEATURE_MAP_SIZE,
                      kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (FEATURE_MAP_SIZE) x 32 x 32
            nn.Conv2d(in_channels = self.FEATURE_MAP_SIZE,
                      out_channels = self.FEATURE_MAP_SIZE * 2,
                      kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(self.FEATURE_MAP_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (FEATURE_MAP_SIZE*2) x 16 x 16
            nn.Conv2d(in_channels = self.FEATURE_MAP_SIZE * 2,
                      out_channels = self.FEATURE_MAP_SIZE * 4,
                      kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(self.FEATURE_MAP_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (FEATURE_MAP_SIZE*4) x 8 x 8'
            nn.Conv2d(in_channels = self.FEATURE_MAP_SIZE * 4,
                      out_channels = self.FEATURE_MAP_SIZE * 8,
                      kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(self.FEATURE_MAP_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (FEATURE_MAP_SIZE*8) x 4 x 4
            nn.Conv2d(in_channels = self.FEATURE_MAP_SIZE * 8,
                      out_channels = 1,
                      kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, x):
        return self.main(x)

    def init_weights(self):
        self.apply(weights_init)


def plot_losses(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def generate_images(generator, num_images=20):
    with torch.no_grad():
        noise = torch.randn(num_images, generator.INPUT_SIZE, 1, 1,
                            device=next(generator.parameters()).device)
        fake = generator(noise).detach().cpu()
        grid = vutils.make_grid(fake, nrow=10, padding=2, normalize=True)
        plt.figure(figsize=(30, 3))
        plt.axis("off")
        plt.title("Generated Fake Images")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
