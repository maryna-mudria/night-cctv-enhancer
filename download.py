# torchvision CelebA качается с Google Drive и упёрлась в quota
# переключилась на HuggingFace, там тот же CelebA но через нормальный API
# сохраняем первые N картинок 256x256 в data/faces/

import os
import cv2
import numpy as np
from datasets import load_dataset

OUT = "data/faces"
N = 5000
SIZE = 256

os.makedirs(OUT, exist_ok=True)

print("качаю celebA с HuggingFace (streaming)")
ds = load_dataset("huggan/CelebA-faces", split="train", streaming=True)

i = 0
for item in ds:
    if i >= N:
        break
    img = item["image"]            # PIL RGB
    img = img.resize((SIZE, SIZE))
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT, str(i).zfill(6) + ".jpg"), bgr)
    i += 1
    if i % 500 == 0:
        print(i)

print("готово:", i)
