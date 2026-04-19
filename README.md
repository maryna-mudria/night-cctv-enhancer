# Night CCTV Enhancer

Курсовой проект по курсу Computer Vision (Robot Dreams).

## Идея

Камеры видеонаблюдения часто выдают картинку плохого качества — особенно ночью: низкое разрешение, шум, артефакты от JPEG-сжатия. Если на такой кадр попало лицо или номер машины — распознать ничего не получится.

Делаю модель, которая одновременно:
- убирает шум (denoising)
- повышает разрешение в 4 раза (super-resolution)

И сравниваю её с классическими `cv2.resize` + `cv2.GaussianBlur`.

## Датасет

Хотела взять CelebA (лица), но её и LFW убрали из `torchvision.datasets` (проблемы с лицензиями). Переключилась на **Oxford-IIIT Pet** — 37 пород котов и собак, ~7400 картинок. Для super-resolution контент не важен, сеть учится восстанавливать детали текстуры.

Беру 5000 картинок, режу центр-кропом до 256×256. Папка так и называется `data/faces/`, чтобы не править остальной код.

## Пайплайн порчи (corruption)

Каждую картинку на лету превращаю из HR в LR:

1. `downsample×4` — bicubic-уменьшение 256→64
2. `motion blur` с вероятностью 0.3 (имитация движения)
3. Гауссов шум, σ = 15–30
4. JPEG q = 30–50 (блочные артефакты)

Таким образом сеть видит пары `(lr 64×64, hr 256×256)`.

## Архитектура

Простой U-Net (3 уровня + bottleneck), как в лекции 16 по сегментации:
- encoder: Conv×2 → MaxPool, каналы 32 → 64 → 128
- bottleneck: 256
- decoder: bilinear upsample → concat skip → Conv×2
- выход: `sigmoid` в [0, 1]

Перед входом в encoder картинку сначала увеличиваю bicubic-ом до 256×256, чтобы сеть сразу работала в полном разрешении и занималась тем, что умеет — чистила шум и добавляла детали.

## Бейзлайны

Сравниваю с тремя классическими способами (все в `baselines.py`):
- `bicubic` — просто `cv2.INTER_CUBIC`
- `gaussian+bicubic` — сначала Gaussian blur 3×3, потом bicubic
- `bilateral+bicubic` — сначала bilateral (сохраняет границы), потом bicubic

## Метрики

- **PSNR** — peak signal-to-noise ratio, чем больше, тем лучше (дБ)
- **SSIM** — structural similarity, от 0 до 1

Обе беру из `skimage.metrics`.

## Как запускать

```
pip install -r requirements.txt

# 1) скачать датасет (~5000 картинок 256x256)
python download.py

# 2) sanity check перед полным обучением
#    проверяет что сеть вообще способна выучить задачу
python sanity_check.py

# 3) полное обучение (10 эпох, сохраняет лучшую модель в models/unet.pth)
#    параллельно можно смотреть кривые в TensorBoard:
#    tensorboard --logdir runs
python train.py

# 4) оценка — считает PSNR/SSIM для всех бейзлайнов и модели,
#    сохраняет наглядные примеры в samples/
python evaluate.py
```

## Структура проекта

```
download.py       — скачивание и подготовка датасета
corruption.py     — функции порчи (downsample, noise, JPEG, blur)
dataset.py        — PyTorch Dataset, отдаёт пары (lr, hr)
model.py          — U-Net
baselines.py      — классические методы для сравнения
metrics.py        — PSNR, SSIM
sanity_check.py   — overfit на одном батче для проверки кода
train.py          — обучение + TensorBoard + save-best
evaluate.py       — финальная оценка и примеры
requirements.txt  — зависимости
```

## Статус

- [x] подготовка данных
- [x] corruption-пайплайн
- [x] U-Net
- [x] бейзлайны и метрики
- [x] sanity-check прошёл (код рабочий)
- [ ] полное обучение
- [ ] финальная оценка и таблица результатов
- [ ] презентация
