# CelebA не качается вообще: гугл дал quota, а с huggingface датасет убрали
# пробую LFW (Labeled Faces in the Wild), он тоже из лиц и лежит на universe-серверах
# первые N картинок сохраняем 256x256 в data/faces/

import os
import cv2
import numpy as np
from torchvision import datasets, transforms

OUT = "data/faces"
N = 5000
SIZE = 256

os.makedirs(OUT, exist_ok=True)

# resize по короткой стороне до SIZE и центр-кроп квадрат
tfm = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.CenterCrop(SIZE),
])

print("качаю LFW")
ds = datasets.LFWPeople(root="data/lfw_raw", download=True, transform=tfm)
print("всего картинок в LFW:", len(ds))

n = min(N, len(ds))
print("сохраняем", n)

for i in range(n):
    img, _ = ds[i]            # PIL, может быть grayscale
    arr = np.array(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT, str(i).zfill(6) + ".jpg"), bgr)
    if i % 500 == 0:
        print(i)

print("готово")
