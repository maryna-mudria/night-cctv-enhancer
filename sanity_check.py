# sanity check перед полным обучением
# идея: взять маленький батч и переобучить на нём модель за много эпох
# если loss падает почти до нуля — значит сеть вообще способна выучить задачу
# и код без багов. если нет — лучше это узнать сейчас, а не после ночи обучения
# (Ян на лекции 17 так и говорил — sanity check на одном батче)

import os
import random
import numpy as np
import torch
import torch.nn as nn
import cv2

from dataset import CCTVFaces
from model import UNet


DATA_DIR = "data/faces"
BATCH = 8
EPOCHS = 200
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# фиксируем все сиды, иначе результат будет плясать
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("device:", DEVICE)

# берём один маленький батч из датасета и запоминаем его
# (важно — не генерить заново каждую эпоху, иначе это уже не "один батч")
full = CCTVFaces(DATA_DIR)
print("всего картинок:", len(full))

lrs, hrs = [], []
for i in range(BATCH):
    lr_t, hr_t = full[i]
    lrs.append(lr_t)
    hrs.append(hr_t)
lr_batch = torch.stack(lrs).to(DEVICE)
hr_batch = torch.stack(hrs).to(DEVICE)
print("батч:", lr_batch.shape, "->", hr_batch.shape)

# модель, оптимизатор, loss
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# обучаем на одном и том же батче много эпох
print("начинаем overfit на", BATCH, "картинках за", EPOCHS, "эпох")
for epoch in range(EPOCHS):
    model.train()
    pred = model(lr_batch)
    loss = criterion(pred, hr_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # печатаем каждые 10 эпох чтобы не засорять вывод
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:3d}  loss {loss.item():.6f}")

# если финальный loss < 0.001 — значит сеть способна переобучиться
# и код работает как надо
final = loss.item()
print()
print("финальный loss:", round(final, 6))
if final < 0.001:
    print("ок, модель способна выучить батч — можно запускать полное обучение")
else:
    print("loss не упал как ожидалось, проверь код")

# сохраним картинку с результатом чтобы глазами посмотреть
# должны увидеть почти точное совпадение pred и hr
os.makedirs("samples", exist_ok=True)
model.eval()
with torch.no_grad():
    pred = model(lr_batch)

# берём первую картинку из батча, склеиваем lr (увеличенный до 256) | pred | hr
lr_big = torch.nn.functional.interpolate(
    lr_batch[:1], scale_factor=4, mode="bicubic", align_corners=False
).clamp(0, 1)

imgs = [lr_big[0], pred[0].clamp(0, 1), hr_batch[0]]
imgs = [(t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8) for t in imgs]
grid = np.hstack(imgs)
bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
cv2.imwrite("samples/sanity_check.png", bgr)
print("картинка сохранена в samples/sanity_check.png (lr-bicubic | pred | hr)")
