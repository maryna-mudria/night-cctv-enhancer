# U-Net чтобы одновременно убирать шум и увеличивать разрешение картинки
# вход: LR 64x64 (испорченная картинка)
# выход: HR 256x256 (восстановленная)
# архитектура примерно как в лекции 16, только подогнала под размер 256

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # encoder
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)

        # bottleneck
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)

        # decoder (размерности склеиваем с соответствующими skip connections)
        self.conv_u3_1 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.conv_u3_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv_u2_1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.conv_u2_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_u1_1 = nn.Conv2d(64 + 32, 32, 3, padding=1)
        self.conv_u1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        # сначала увеличиваем картинку bicubic'ом до нужного размера
        # а сеть пусть подчистит шум и добавит детали
        x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        x = torch.clamp(x, 0, 1)

        # encoder
        c1 = F.relu(self.conv1_1(x))
        c1 = F.relu(self.conv1_2(c1))

        c2 = F.relu(self.conv2_1(self.pool(c1)))
        c2 = F.relu(self.conv2_2(c2))

        c3 = F.relu(self.conv3_1(self.pool(c2)))
        c3 = F.relu(self.conv3_2(c3))

        # bottleneck
        c4 = F.relu(self.conv4_1(self.pool(c3)))
        c4 = F.relu(self.conv4_2(c4))

        # decoder со skip connections
        u3 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = F.relu(self.conv_u3_1(u3))
        u3 = F.relu(self.conv_u3_2(u3))

        u2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = F.relu(self.conv_u2_1(u2))
        u2 = F.relu(self.conv_u2_2(u2))

        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = F.relu(self.conv_u1_1(u1))
        u1 = F.relu(self.conv_u1_2(u1))

        # sigmoid чтобы выход был в [0, 1] как и вход
        return torch.sigmoid(self.out(u1))
