# скрипт чтобы скачать датасет CelebA и сохранить первые N картинок
# размером 256х256 в data/faces/

import os
import numpy as np
import cv2
from torchvision import datasets, transforms

OUT = "data/faces"
N = 5000        # сколько картинок берём
SIZE = 256

os.makedirs(OUT, exist_ok=True)

# сначала resize по короткой стороне, потом центр-кроп
tfm = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.CenterCrop(SIZE),
])

ds = datasets.CelebA(root="data/celeba_raw", split="train",
                     download=True, transform=tfm)
print("всего в CelebA train:", len(ds))
print("сохраняем первые", N)

for i in range(N):
    img, _ = ds[i]                # img это PIL
    arr = np.array(img)           # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    name = str(i).zfill(6) + ".jpg"
    cv2.imwrite(os.path.join(OUT, name), bgr)
    if i % 500 == 0:
        print(i)

print("готово")
