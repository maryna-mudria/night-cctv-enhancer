# простые "классические" бейзлайны которые сравниваем с моделью
# берём маленькую плохую картинку и пытаемся получить большую хорошую

import cv2


def bicubic(lr, size=256):
    # просто increase без denoise
    return cv2.resize(lr, (size, size), interpolation=cv2.INTER_CUBIC)


def gaussian_bicubic(lr, size=256):
    # сначала сгладим шум гауссом, потом увеличим
    blur = cv2.GaussianBlur(lr, (3, 3), 1.0)
    return cv2.resize(blur, (size, size), interpolation=cv2.INTER_CUBIC)


def bilateral_bicubic(lr, size=256):
    # bilateral сохраняет границы в отличие от gaussian
    blur = cv2.bilateralFilter(lr, 9, 75, 75)
    return cv2.resize(blur, (size, size), interpolation=cv2.INTER_CUBIC)
