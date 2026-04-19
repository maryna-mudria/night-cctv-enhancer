# скрипт обучения модели
# добавила TensorBoard чтобы видеть train/val loss на графике
# и save-best — сохраняем модель только когда val_loss улучшился
# (Ян рассказывал байку как он обучал сеть 10 дней и забыл сохранить веса)

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import CCTVFaces
from model import UNet


DATA_DIR = "data/faces"
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# фиксируем все сиды чтобы результаты были воспроизводимы
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("device:", DEVICE)

# загружаем датасет и делим на train / val
full = CCTVFaces(DATA_DIR)
print("всего картинок:", len(full))

val_size = int(len(full) * 0.1)
train_size = len(full) - val_size
train_ds, val_ds = random_split(full, [train_size, val_size])

# num_workers=0 потому что на винде ругается на мультипроцессинг без if __name__ == "__main__"
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# модель и оптимизатор
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# TensorBoard — логи пишутся в runs/unet
# смотреть: tensorboard --logdir runs
os.makedirs("runs", exist_ok=True)
writer = SummaryWriter("runs/unet")

# готовим папку для весов и трекаем лучший val_loss
os.makedirs("models", exist_ok=True)
best_val = float("inf")

# цикл обучения
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    count = 0
    for lr, hr in train_loader:
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        pred = model(lr)
        loss = criterion(pred, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * lr.size(0)
        count += lr.size(0)

    avg_train = total / count

    # валидация
    model.eval()
    total_v = 0.0
    count_v = 0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            pred = model(lr)
            total_v += criterion(pred, hr).item() * lr.size(0)
            count_v += lr.size(0)
    avg_val = total_v / count_v

    # логируем в TensorBoard
    writer.add_scalar("loss/train", avg_train, epoch)
    writer.add_scalar("loss/val", avg_val, epoch)

    print("epoch", epoch, "train_loss", round(avg_train, 4), "val_loss", round(avg_val, 4))

    # сохраняем только если это лучший val_loss за всё время
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), "models/unet.pth")
        print("  -> новый лучший val_loss, сохранила в models/unet.pth")

writer.close()
print("обучение закончено, лучший val_loss:", round(best_val, 4))
