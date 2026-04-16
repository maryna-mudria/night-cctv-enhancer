# метрики чтобы считать насколько хорошо модель восстановила картинку
# PSNR - peak signal to noise ratio (в дБ, чем больше тем лучше)
# SSIM - structural similarity (от 0 до 1)

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def psnr(pred, target):
    return peak_signal_noise_ratio(target, pred, data_range=255)


def ssim(pred, target):
    return structural_similarity(target, pred, data_range=255, channel_axis=-1)
