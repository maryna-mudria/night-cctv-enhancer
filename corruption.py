# функции которые "портят" картинку чтобы она была похожа на
# кадр с обычной IP камеры ночью: низкое разрешение и шум

import cv2
import numpy as np


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
