"""KVCache — эффективный менеджер KV-кэша для трансформеров.

Ключевая идея: вместо np.concatenate на каждом токене (что даёт
квадратичную сложность), мы заранее выделяем буфер максимального размера
и записываем новые K/V в срез. Чтение возвращает view, а не копию.

Публичный API:
    cache = KVCache(num_layers, num_kv_heads, head_dim, max_seq_len)
    cache.append(layer_idx, k_new, v_new)
    k, v = cache.get(layer_idx)
    cache.reset()
"""
import numpy as np


class KVCache:
    """Менеджер KV-кэша с предвыделенной памятью.

    Параметры:
        num_layers: количество слоёв
        num_kv_heads: количество KV-голов (для GQA/MQA меньше, чем Q-головы)
        head_dim: размер одной головы
        max_seq_len: максимальная длина последовательности (контекст)
        batch_size: размер батча (пока 1)
        dtype: dtype для хранения (внутри всегда float32 для скорости)
    """

    def __init__(self, num_layers, num_kv_heads, head_dim,
                 max_seq_len, batch_size=1, dtype=np.float32):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype

        shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)

        self.cur_len = 0

    def append(self, layer_idx, k_new, v_new):
        """Записывает новые K/V для одного слоя.

        Параметры:
            layer_idx: индекс слоя (0 .. num_layers-1)
            k_new: (batch_size, num_kv_heads, num_new_tokens, head_dim)
            v_new: то же для V
        """
        n_new = k_new.shape[2]
        pos = self.cur_len
        if pos + n_new > self.max_seq_len:
            raise ValueError(
                f"KV-кэш переполнен: {pos + n_new} > {self.max_seq_len}. "
                f"Увеличьте max_seq_len или используйте sliding window."
            )
        self.k[layer_idx, :, :, pos:pos + n_new, :] = k_new
        self.v[layer_idx, :, :, pos:pos + n_new, :] = v_new

    def get(self, layer_idx):
        """Возвращает (k, v) для слоя. Возвращает view, а не копию."""
        return (
            self.k[layer_idx, :, :, :self.cur_len, :],
            self.v[layer_idx, :, :, :self.cur_len, :],
        )

    def advance(self, n=1):
        """Продвигает счётчик длины после записи всех слоёв."""
        self.cur_len += n

    def reset(self):
        """Полностью сбрасывает кэш (новый запрос)."""
        self.cur_len = 0

    def memory_usage_bytes(self):
        """Сколько байт занимает кэш."""
        return self.k.nbytes + self.v.nbytes

    def memory_usage_human(self):
        """Читаемый размер."""
        n = self.memory_usage_bytes()
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} TB"

    def __repr__(self):
        return (
            f"<KVCache layers={self.num_layers} heads={self.num_kv_heads} "
            f"head_dim={self.head_dim} len={self.cur_len}/{self.max_seq_len} "
            f"mem={self.memory_usage_human()}>"
        )