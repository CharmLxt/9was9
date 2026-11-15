"""RoPECache — предвычисленные sin/cos таблицы для Rotary Position Embeddings.

RoPE применяется в каждой голове каждого слоя к каждому токену. Наивный
подход — считать sin/cos на лету — тратит до 15% времени. Мы предвычисляем
их один раз для максимальной длины контекста и кэшируем.

Поддерживаемые варианты:
- linear  (rope_scaling: type="linear", factor=F)
- dynamic (rope_scaling: type="dynamic", factor=F)
- llama3  (rope_scaling: type="llama3", low_freq_factor, high_freq_factor,
           original_max_position_embeddings)

Публичный API:
    cache = RoPECache(head_dim, max_seq_len, rope_theta=10000.0,
                      rope_scaling=None)
    cos, sin = cache.get(seq_len)  # (seq_len, head_dim)
    q_rot, k_rot = RoPECache.apply(q, k, cos, sin)
"""
import numpy as np


class RoPECache:
    """Предвычисленные cos/sin для RoPE.

    Параметры:
        head_dim: размер одной головы (должен быть чётным)
        max_seq_len: максимальная длина последовательности
        rope_theta: базовая частота (обычно 10000.0)
        rope_scaling: dict с настройками или None
    """

    def __init__(self, head_dim, max_seq_len, rope_theta=10000.0,
                 rope_scaling=None, dtype=np.float32):
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim должен быть чётным, получено {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {}
        self.dtype = dtype

        # Частоты: 1 / theta^(2i/head_dim), i = 0..head_dim/2-1
        half_dim = head_dim // 2
        freqs = 1.0 / (rope_theta ** (np.arange(0, half_dim, dtype=np.float64) / half_dim))

        # Применяем scaling
        scaling_type = self.rope_scaling.get("type", "linear")
        if scaling_type == "linear":
            factor = self.rope_scaling.get("factor", 1.0)
            freqs = freqs / factor
        elif scaling_type == "dynamic":
            # Динамическое масштабирование (NTK-aware). Упрощённая версия:
            # тот же линейный скейл, но по факту применяется только к длинным
            # последовательностям. Для простоты делаем как linear.
            factor = self.rope_scaling.get("factor", 1.0)
            freqs = freqs / factor
        elif scaling_type == "llama3":
            # Llama 3 style scaling
            factor = self.rope_scaling.get("factor", 8.0)
            low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0)
            orig_max_pos = self.rope_scaling.get("original_max_position_embeddings", 8192)

            low_freq_wavelen = orig_max_pos / low_freq_factor
            high_freq_wavelen = orig_max_pos / high_freq_factor

            wavelens = 2 * np.pi / freqs

            # Низкие частоты (длинные волны) — делим на фактор
            freqs = np.where(
                wavelens > low_freq_wavelen,
                freqs / factor,
                freqs,
            )
            # Высокие частоты (короткие волны) — оставляем
            # Промежуточные — плавный переход
            medium_mask = (wavelens <= low_freq_wavelen) & (wavelens >= high_freq_wavelen)
            smooth = (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
            freqs = np.where(
                medium_mask,
                freqs / (factor * (1 - smooth) + smooth),
                freqs,
            )
        # иначе (тип неизвестен) — игнорируем scaling

        # Позиции: 0..max_seq_len-1
        positions = np.arange(max_seq_len, dtype=np.float64)
        # Внешнее произведение: (max_seq_len, half_dim)
        angles = np.outer(positions, freqs)

        cos_half = np.cos(angles).astype(dtype)
        sin_half = np.sin(angles).astype(dtype)

        # Дублируем на весь head_dim: [cos, cos] и [sin, sin]
        self.cos = np.concatenate([cos_half, cos_half], axis=-1)  # (L, D)
        self.sin = np.concatenate([sin_half, sin_half], axis=-1)  # (L, D)

    def get(self, seq_len):
        """Возвращает (cos, sin) для первых seq_len позиций.

        Возвращает:
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} > max_seq_len={self.max_seq_len}"
            )
        return self.cos[:seq_len], self.sin[:seq_len]

    def slice_for_positions(self, start, end):
        """Возвращает cos/sin для позиций [start, end)."""
        return self.cos[start:end], self.sin[start:end]

    @staticmethod
    def rotate_half(x):
        """Поворот половины: [-x2, x1] для x = [x1, x2]."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return np.concatenate([-x2, x1], axis=-1)

    @staticmethod
    def apply(q, k, cos, sin):
        """Применяет RoPE к Q и K.

        Параметры:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)

        Возвращает:
            q_rot, k_rot с той же формой.
        """
        # cos/sin нужно broadcast'ить: (1, 1, seq_len, head_dim)
        cos = cos[np.newaxis, np.newaxis, :, :]
        sin = sin[np.newaxis, np.newaxis, :, :]

        q_rot = (q * cos) + (RoPECache.rotate_half(q) * sin)
        k_rot = (k * cos) + (RoPECache.rotate_half(k) * sin)
        return q_rot, k_rot

    def __repr__(self):
        return (
            f"<RoPECache head_dim={self.head_dim} max_seq={self.max_seq_len} "
            f"theta={self.rope_theta} scaling={self.rope_scaling.get('type', 'none')}>"
        )