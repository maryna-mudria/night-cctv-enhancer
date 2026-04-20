# дообучение v1 модели (models/unet.pth) с новыми настройками
# что поменялось относительно train.py:
# - loss: MSE -> L1 (меньше "мыла" на краях)
# - LR понижен 1e-3 -> 1e-4 (стандарт для fine-tune)
# - добавлен ReduceLROnPlateau(patience=2, factor=0.5)
# - веса стартуют не с нуля, а из unet.pth
# - лучший чекпойнт пишется в unet_v2.pth, чтобы можно было сравнить v1 vs v2
#
# архитектуру не трогала, чтобы load_state_dict прошёл.
# на CPU ~68 мин на эпоху, ставлю 10 эпох = около 11 часов (на ночь).

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from dataset import CCTVFaces
from model import UNet


DATA_DIR = "data/faces"
CKPT_IN = "models/unet.pth"
CKPT_OUT = "models/unet_v2.pth"
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# те же сиды что и в train.py — чтобы split train/val совпал
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("device:", DEVICE)

# датасет и тот же самый split 90/10
full = CCTVFaces(DATA_DIR)
print("всего картинок:", len(full))
val_size = int(len(full) * 0.1)
train_size = len(full) - val_size
train_ds, val_ds = random_split(full, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# модель + загрузка v1 весов
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT_IN, map_location=DEVICE))
print("стартуем с:", CKPT_IN)

# L1 loss вместо MSE + пониженный LR
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()

# scheduler: режем LR пополам если 2 эпохи подряд val_loss не улучшается
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

# TensorBoard -> runs/unet_v2
os.makedirs("runs", exist_ok=True)
writer = SummaryWriter("runs/unet_v2")

os.makedirs("models", exist_ok=True)
best_val = float("inf")

# оценим val_loss до обучения — это точка отсчёта
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
best_val = total_v / count_v
print("стартовый val_loss (L1):", round(best_val, 5))
writer.add_scalar("loss/val", best_val, -1)

# цикл fine-tune
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

    writer.add_scalar("loss/train", avg_train, epoch)
    writer.add_scalar("loss/val", avg_val, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    cur_lr = optimizer.param_groups[0]["lr"]
    print("epoch", epoch,
          "train_loss", round(avg_train, 5),
          "val_loss", round(avg_val, 5),
          "lr", cur_lr)

    # save-best
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), CKPT_OUT)
        print("  -> новый лучший val_loss, сохранила в", CKPT_OUT)

    # scheduler шагает по val_loss
    scheduler.step(avg_val)

writer.close()
print("fine-tune закончен, лучший val_loss:", round(best_val, 5))
