# сравниваем нашу модель с классическими методами
# считаем PSNR и SSIM + сохраняем пару наглядных примеров
#
# три inference-time доработки (без переобучения):
# 1) TTA — прогоняем сеть на картинке и на её горизонтальном флипе,
#    усредняем. Идея аугментаций из лекции 15, только применяется при оценке.
# 2) α-blending с bilateral+bicubic — линейно смешиваем выход сети
#    с лучшим классическим бейзлайном. UNet после MSE даёт "мыло",
#    bilateral+bicubic сохраняет резкие границы — смесь вытягивает PSNR/SSIM.
# 3) USM (unsharp masking) поверх blend — классика из лекции 3.
#    После blend шум уже убран сетью, значит USM чисто добавит резкости.

import os
import random
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
N_SAMPLES = 5        # сколько наглядных примеров сохраним

# сетка α для смешивания UNet+TTA с bilateral+bicubic
# α=1.0 — чистая сеть, α=0.0 — чистый бейзлайн
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# сетка amount для USM. 0.0 = без USM. cv2 USM через gaussian blur.
USM_AMOUNTS = [0.0, 0.3, 0.5, 0.7, 1.0]

# фиксируем ВСЕ сиды — не только torch.
# в corruption.py используется random и np.random, без их сидов
# между прогонами результаты плавают на ±0.1 дБ.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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


def unet_tta(lr_t):
    # обычный проход + проход по отражённой картинке, усреднение
    inp = lr_t.unsqueeze(0).to(device)
    out1 = model(inp)
    inp_flip = torch.flip(inp, dims=[-1])
    out2 = model(inp_flip)
    out2 = torch.flip(out2, dims=[-1])
    out = (out1 + out2) / 2
    return (out[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def unsharp_mask(img, amount, sigma=1.0, ksize=5):
    # классический USM: sharp = img + amount * (img - blur)
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    sharp = img.astype(np.float32) + amount * (img.astype(np.float32) - blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


# копим метрики
results = {
    "bicubic": {"psnr": [], "ssim": []},
    "gaussian+bicubic": {"psnr": [], "ssim": []},
    "bilateral+bicubic": {"psnr": [], "ssim": []},
    "UNet": {"psnr": [], "ssim": []},
    "UNet+TTA": {"psnr": [], "ssim": []},
}
# комбинации (α, amount) для blend + USM
for a in ALPHAS:
    for m in USM_AMOUNTS:
        results[f"blend a={a:.1f} usm={m:.1f}"] = {"psnr": [], "ssim": []}

n = len(val_ds)  # все val-картинки для стабильных цифр
print("оцениваем на", n, "картинках")

with torch.no_grad():
    for i in range(n):
        lr_t, hr_t = val_ds[i]
        # в numpy uint8 для метрик и для бейзлайнов
        lr_np = (lr_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        hr_np = (hr_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # бейзлайны
        b = bicubic(lr_np)
        results["bicubic"]["psnr"].append(psnr(b, hr_np))
        results["bicubic"]["ssim"].append(ssim(b, hr_np))

        gb = gaussian_bicubic(lr_np)
        results["gaussian+bicubic"]["psnr"].append(psnr(gb, hr_np))
        results["gaussian+bicubic"]["ssim"].append(ssim(gb, hr_np))

        bb = bilateral_bicubic(lr_np)
        results["bilateral+bicubic"]["psnr"].append(psnr(bb, hr_np))
        results["bilateral+bicubic"]["ssim"].append(ssim(bb, hr_np))

        # UNet — обычный проход
        inp = lr_t.unsqueeze(0).to(device)
        out = model(inp)
        pred_np = (out[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        results["UNet"]["psnr"].append(psnr(pred_np, hr_np))
        results["UNet"]["ssim"].append(ssim(pred_np, hr_np))

        # UNet + TTA
        tta_np = unet_tta(lr_t)
        results["UNet+TTA"]["psnr"].append(psnr(tta_np, hr_np))
        results["UNet+TTA"]["ssim"].append(ssim(tta_np, hr_np))

        # все комбинации (α, USM amount)
        for a in ALPHAS:
            blend = a * tta_np.astype(np.float32) + (1 - a) * bb.astype(np.float32)
            blend = np.clip(blend, 0, 255).astype(np.uint8)
            for m in USM_AMOUNTS:
                out_np = unsharp_mask(blend, m)
                key = f"blend a={a:.1f} usm={m:.1f}"
                results[key]["psnr"].append(psnr(out_np, hr_np))
                results[key]["ssim"].append(ssim(out_np, hr_np))

# считаем средние
means = {name: (float(np.mean(r["psnr"])), float(np.mean(r["ssim"])))
         for name, r in results.items()}

# ищем лучшую (α, USM amount) пару по PSNR
combo_keys = [k for k in means if k.startswith("blend ")]
best_key = max(combo_keys, key=lambda k: means[k][0])
best_a = float(best_key.split("a=")[1].split()[0])
best_m = float(best_key.split("usm=")[1])

# печатаем таблицу по секциям
print()
print("результаты (среднее по", n, "картинкам):")
print("метод                             PSNR     SSIM")

# секция 1: бейзлайны и сеть
for name in ["bicubic", "gaussian+bicubic", "bilateral+bicubic", "UNet", "UNet+TTA"]:
    p, s = means[name]
    print(f"{name:33s} {p:6.2f}   {s:.4f}")

# секция 2: α-sweep без USM (показываем что даёт один blend)
print()
print("blend с bilateral+bicubic, без USM:")
for a in ALPHAS:
    name = f"blend a={a:.1f} usm=0.0"
    p, s = means[name]
    print(f"  a={a:.1f}                          {p:6.2f}   {s:.4f}")

# секция 3: USM sweep на лучшем α
print()
print(f"USM поверх blend a={best_a:.1f}:")
for m in USM_AMOUNTS:
    name = f"blend a={best_a:.1f} usm={m:.1f}"
    p, s = means[name]
    marker = "  <-- лучшее" if name == best_key else ""
    print(f"  usm={m:.1f}                        {p:6.2f}   {s:.4f}{marker}")

print()
print(f"лучшая комбинация: a={best_a:.1f}, usm={best_m:.1f}"
      f" -> PSNR {means[best_key][0]:.2f}, SSIM {means[best_key][1]:.4f}")

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
        bb = bilateral_bicubic(lr_np)

        tta_np = unet_tta(lr_t)
        blend = np.clip(best_a * tta_np.astype(np.float32) + (1 - best_a) * bb.astype(np.float32),
                        0, 255).astype(np.uint8)
        final = unsharp_mask(blend, best_m)

        # слева направо:
        # bicubic | bilateral+bicubic | UNet+TTA | UNet+TTA+blend+USM | оригинал
        grid = np.hstack([b, bb, tta_np, final, hr_np])
        bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"samples/example_{i}.png", bgr)

print("готово")
