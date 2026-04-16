# класс который отдаёт модели пары (испорченная картинка, оригинал)
# испорченная версия генерируется на лету функцией corrupt()

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from corruption import corrupt


class CCTVFaces(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(os.listdir(folder))
        # фильтруем только картинки
        self.files = [f for f in self.files if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # hr это оригинал 256x256, lr это тот же снимок после corrupt
        hr = img
        lr = corrupt(hr)

        # превращаем в тензоры вида (C, H, W) и нормализуем в [0, 1]
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0

        return lr_t, hr_t
