# функции которые "портят" картинку чтобы она была похожа на
# кадр с обычной IP камеры ночью: низкое разрешение, шум, jpeg-артефакты,
# смаз от движения

import cv2
import numpy as np
import random
from io import BytesIO
from PIL import Image


def downsample(img, scale=4):
    # уменьшаем картинку в scale раз
    h, w = img.shape[:2]
    new_w = w // scale
    new_h = h // scale
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def add_gaussian_noise(img, sigma):
    # добавляем гауссовский шум
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def add_jpeg(img, quality):
    # сохраняем в jpeg с низким качеством и читаем обратно
    # чтобы получить jpeg-артефакты (блоки 8x8)
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))


def motion_blur(img, size=5):
    # размытие от движения, просто прямая линия в ядре
    k = np.zeros((size, size), dtype=np.float32)
    angle = random.uniform(0, 180)
    c = size // 2
    k[c, :] = 1.0 / size
    M = cv2.getRotationMatrix2D((c, c), angle, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    k /= k.sum() + 1e-8
    return cv2.filter2D(img, -1, k)


def corrupt(img):
    # весь пайплайн: уменьшаем -> иногда смаз -> шум -> jpeg
    out = downsample(img, 4)
    # смаз добавляем не всегда (бывают же и неподвижные сцены)
    if random.random() < 0.3:
        k = random.choice([3, 5, 7])
        out = motion_blur(out, k)
    sigma = random.uniform(15, 30)
    out = add_gaussian_noise(out, sigma)
    q = random.randint(30, 50)
    out = add_jpeg(out, q)
    return out
