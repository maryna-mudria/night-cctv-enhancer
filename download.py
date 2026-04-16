# LFW тоже удалили из torchvision... беру Oxford-IIIT Pet
# 37 пород котов и собак, ~7400 картинок, стабильно качается из университета
# для super-resolution контент неважен, модель учится восстанавливать детали текстуры

import os
import cv2
import numpy as np
from torchvision import datasets, transforms

OUT = "data/faces"           # оставляю старое имя папки чтобы не править dataset.py
N = 5000
SIZE = 256

os.makedirs(OUT, exist_ok=True)

# resize по короткой стороне + центр-кроп до квадрата
tfm = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.CenterCrop(SIZE),
])

print("качаю Oxford Pets")
ds = datasets.OxfordIIITPet(root="data/pets_raw", download=True, transform=tfm)
print("всего картинок:", len(ds))

n = min(N, len(ds))
print("сохраняем", n)

for i in range(n):
    img, _ = ds[i]
    arr = np.array(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT, str(i).zfill(6) + ".jpg"), bgr)
    if i % 500 == 0:
        print(i)

print("готово")
