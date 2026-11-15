"""fast_softmax — numerically stable softmax и scaled dot-product attention."""
import numpy as np


def fast_softmax(x, axis=-1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    exp_x = np.exp(x - x_max)
    denom = np.sum(exp_x, axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return exp_x / denom


def log_softmax(x, axis=-1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    x_shifted = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True) + 1e-12)
    return x_shifted - log_sum_exp


def apply_causal_mask(scores, seq_len=None):
    if seq_len is None:
        seq_len = scores.shape[-1]
    mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
    return scores + mask


def scaled_dot_product_attention(q, k, v, mask=None, scale=None, dropout_p=0.0):
    """Scaled dot-product attention. GQA/MQA через broadcast (без np.repeat).

    q: (batch, num_heads_q, seq_q, head_dim)
    k: (batch, num_kv_heads, seq_kv, head_dim)
    v: (batch, num_kv_heads, seq_kv, head_dim)
    """
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    num_heads_q = q.shape[1]
    num_kv_heads = k.shape[1]

    if num_kv_heads == num_heads_q:
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        attn_weights = fast_softmax(scores, axis=-1)
        if dropout_p > 0:
            keep = np.random.rand(*attn_weights.shape) > dropout_p
            attn_weights = attn_weights * keep / (1.0 - dropout_p)
        return np.matmul(attn_weights, v)

    if num_heads_q % num_kv_heads != 0:
        raise ValueError(
            f"num_heads_q={num_heads_q} не делится на num_kv_heads={num_kv_heads}"
        )

    repeat_factor = num_heads_q // num_kv_heads
    batch = q.shape[0]
    seq_q = q.shape[2]
    head_dim = q.shape[3]
    seq_kv = k.shape[2]

    q_exp = q.reshape(batch, num_kv_heads, repeat_factor, seq_q, head_dim)
    k_exp = k[:, :, None, :, :]
    v_exp = v[:, :, None, :, :]

    scores = np.matmul(q_exp, np.swapaxes(k_exp, -1, -2)) * scale
    if mask is not None:
        scores = scores + mask[None, None, None, :, :]

    attn_weights = fast_softmax(scores, axis=-1)
    if dropout_p > 0:
        keep = np.random.rand(*attn_weights.shape) > dropout_p
        attn_weights = attn_weights * keep / (1.0 - dropout_p)

    out = np.matmul(attn_weights, v_exp)
    return out.reshape(batch, num_heads_q, seq_q, head_dim)


def top_k_filter(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    thresh = np.partition(logits, -k, axis=-1)[..., -k][..., np.newaxis]
    return np.where(logits >= thresh, logits, -np.inf)


def top_p_filter(logits, p):
    if p >= 1.0:
        return logits
    sorted_indices = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = fast_softmax(sorted_logits, axis=-1)
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)
    sorted_mask = cumulative_probs - sorted_probs >= p
    sorted_logits = np.where(sorted_mask, -np.inf, sorted_logits)
    result = np.empty_like(logits)
    np.put_along_axis(result, sorted_indices, sorted_logits, axis=-1)
    return result


__all__ = [
    "fast_softmax", "log_softmax",
    "apply_causal_mask", "scaled_dot_product_attention",
    "top_k_filter", "top_p_filter",
]