# сравниваем нашу модель с классическими методами
# считаем PSNR и SSIM + сохраняем пару наглядных примеров

import os
import cv2
import numpy as np
import torch
from torch.utils.data import random_split

from dataset import CCTVFaces
from model import UNet
from baselines import bicubic, gaussian_bicubic, bilateral_bicubic
from metrics import psnr, ssim


DATA_DIR = "data/faces"
CKPT = "models/unet.pth"
N_EVAL = 200         # сколько картинок используем для метрик
N_SAMPLES = 5        # сколько наглядных примеров сохраним

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# такая же разбивка на train/val как в train.py
full = CCTVFaces(DATA_DIR)
val_size = int(len(full) * 0.1)
train_size = len(full) - val_size
_, val_ds = random_split(full, [train_size, val_size])
print("картинок в val:", len(val_ds))

# модель
model = UNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# копим метрики
results = {
    "bicubic": {"psnr": [], "ssim": []},
    "gaussian+bicubic": {"psnr": [], "ssim": []},
    "bilateral+bicubic": {"psnr": [], "ssim": []},
    "UNet": {"psnr": [], "ssim": []},
}

n = min(N_EVAL, len(val_ds))
print("оцениваем на", n, "картинках")

with torch.no_grad():
    for i in range(n):
        lr_t, hr_t = val_ds[i]
        # в numpy uint8 для метрик и для бейзлайнов
        lr_np = (lr_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        hr_np = (hr_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # бейзлайны
        pred = bicubic(lr_np)
        results["bicubic"]["psnr"].append(psnr(pred, hr_np))
        results["bicubic"]["ssim"].append(ssim(pred, hr_np))

        pred = gaussian_bicubic(lr_np)
        results["gaussian+bicubic"]["psnr"].append(psnr(pred, hr_np))
        results["gaussian+bicubic"]["ssim"].append(ssim(pred, hr_np))

        pred = bilateral_bicubic(lr_np)
        results["bilateral+bicubic"]["psnr"].append(psnr(pred, hr_np))
        results["bilateral+bicubic"]["ssim"].append(ssim(pred, hr_np))

        # наша модель
        inp = lr_t.unsqueeze(0).to(device)
        out = model(inp)
        pred_np = (out[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        results["UNet"]["psnr"].append(psnr(pred_np, hr_np))
        results["UNet"]["ssim"].append(ssim(pred_np, hr_np))

# таблица
print()
print("результаты (среднее по", n, "картинкам):")
print("метод                    PSNR     SSIM")
for name, r in results.items():
    p = np.mean(r["psnr"])
    s = np.mean(r["ssim"])
    print(f"{name:24s} {p:6.2f}   {s:.4f}")

# наглядные примеры для отчёта
os.makedirs("samples", exist_ok=True)
print()
print("сохраняю", N_SAMPLES, "примеров в samples/")

with torch.no_grad():
    for i in range(N_SAMPLES):
        lr_t, hr_t = val_ds[i]
        lr_np = (lr_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        hr_np = (hr_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        b = bicubic(lr_np)
        gb = gaussian_bicubic(lr_np)

        inp = lr_t.unsqueeze(0).to(device)
        out = model(inp)
        pred_np = (out[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # слева направо: bicubic | gaussian+bicubic | наша модель | оригинал
        grid = np.hstack([b, gb, pred_np, hr_np])
        bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"samples/example_{i}.png", bgr)

print("готово")
