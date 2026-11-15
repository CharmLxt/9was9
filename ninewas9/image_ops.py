"""Быстрые операции с изображениями на чистом numpy (для стриминга и не только)."""
import numpy as np


def fast_resize(img, target_h, target_w):
    """Быстрое изменение размера (Nearest Neighbor).
    
    img: np.ndarray формы (H, W, C) или (H, W).
    target_h, target_w: желаемая высота и ширина.
    """
    h, w = img.shape[:2]
    row_idx = (np.arange(target_h) * (h / target_h)).astype(np.int32)
    col_idx = (np.arange(target_w) * (w / target_w)).astype(np.int32)
    
    if img.ndim == 3:
        return img[row_idx[:, None], col_idx]
    return img[row_idx[:, None], col_idx]


def block_downsample(img, factor):
    """Понижение разрешения усреднением блоков пикселей.
    
    Работает быстрее и дает более гладкую картинку, чем Nearest Neighbor.
    factor: во сколько раз уменьшить (например, 2 для 1080p -> 540p).
    """
    h, w = img.shape[:2]
    # Обрезаем края, если они не делятся на factor
    h_crop = (h // factor) * factor
    w_crop = (w // factor) * factor
    img_cropped = img[:h_crop, :w_crop]
    
    if img.ndim == 3:
        c = img.shape[2]
        reshaped = img_cropped.reshape(h_crop // factor, factor, w_crop // factor, factor, c)
        return reshaped.mean(axis=(1, 3)).astype(img.dtype)
    else:
        reshaped = img_cropped.reshape(h_crop // factor, factor, w_crop // factor, factor)
        return reshaped.mean(axis=(1, 3)).astype(img.dtype)


def rgb_to_grayscale(img):
    """Перевод RGB в оттенки серого.
    
    img: np.ndarray формы (H, W, 3).
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается RGB изображение формы (H, W, 3)")
    # Стандартные веса ITU-R BT.601
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(img.dtype)


def rgb_to_bgr(img):
    """Смена каналов R и B (например, для скриншотов Windows).
    
    img: np.ndarray формы (H, W, 3).
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается RGB изображение формы (H, W, 3)")
    # Меняем первый и третий каналы местами
    return img[..., ::-1]


def normalize_image(img, max_val=255.0):
    """Нормализация пикселей в диапазон [0.0, 1.0].
    
    Полезно, если нужно передать картинку в нейросеть.
    """
    return (img / max_val).astype(np.float32)


def pad_image(img, pad_h, pad_w, fill=0):
    """Добавляет поля к изображению.
    
    img: np.ndarray формы (H, W, C).
    pad_h, pad_w: количество пикселей, добавляемых с каждой стороны.
    """
    if img.ndim == 3:
        h, w, c = img.shape
        padded = np.full((h + 2 * pad_h, w + 2 * pad_w, c), fill, dtype=img.dtype)
        padded[pad_h:pad_h + h, pad_w:pad_w + w, :] = img
        return padded
    else:
        h, w = img.shape
        padded = np.full((h + 2 * pad_h, w + 2 * pad_w), fill, dtype=img.dtype)
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = img
        return padded


__all__ = [
    "fast_resize", "block_downsample", "rgb_to_grayscale", 
    "rgb_to_bgr", "normalize_image", "pad_image"
]