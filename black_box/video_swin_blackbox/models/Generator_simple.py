import torch.nn as nn
import torch.nn.functional as F
import torch


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Generator_simple(nn.Module):
    def __init__(self, z_dim):
        super(Generator_simple, self).__init__()

        final_number = 56 * 56 * 8 * 3

        self.linear = nn.Linear(z_dim, final_number)
        self.flatten = View((-1, 3, 8, 56, 56))
        self.bn0 = nn.BatchNorm3d(3)
        
        self.upsample0 = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv3d(3, 32, (5, 5, 5), padding=(2, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv2 = nn.Conv3d(32, 3, (5, 5, 5), padding=(2, 2, 2))
        self.bn2 = nn.BatchNorm3d(3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.flatten(x)
        x = self.bn0(x)

        x = self.upsample1(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.upsample1(x)

        x = self.conv2(x)
        x_pre = self.bn2(x)
        x = self.tanh(x_pre)
        return x, x_pre
