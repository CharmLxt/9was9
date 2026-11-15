"""Финальный тест ninewas9 0.1.1 — все 5 модулей."""
import sys
import numpy as np
import ninewas9 as nw


print("=" * 60)
print(f"  ninewas9 {nw.__version__} — Full Test")
print("=" * 60)

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  ✅ {name}")
    except Exception as e:
        failed += 1
        print(f"  ❌ {name}: {type(e).__name__}: {e}")


# ============================================================
# 1. KVCache
# ============================================================
print("\n[1] KVCache")


def test_kvcache_basic():
    cache = nw.KVCache(num_layers=2, num_kv_heads=4, head_dim=8, max_seq_len=10)
    assert cache.cur_len == 0
    assert cache.max_seq_len == 10

    k = np.ones((1, 4, 2, 8), dtype=np.float32)
    v = np.ones((1, 4, 2, 8), dtype=np.float32) * 2
    cache.append(0, k, v)
    cache.advance(2)

    k_got = cache.k[0, :, :, :cache.cur_len, :]
    v_got = cache.v[0, :, :, :cache.cur_len, :]
    assert k_got.shape == (1, 4, 2, 8)
    assert np.allclose(k_got, 1.0)
    assert np.allclose(v_got, 2.0)
    assert cache.cur_len == 2


check("KVCache: append + get + advance", test_kvcache_basic)


def test_kvcache_overflow():
    cache = nw.KVCache(num_layers=1, num_kv_heads=2, head_dim=4, max_seq_len=5)
    k = np.zeros((1, 2, 3, 4), dtype=np.float32)
    v = np.zeros((1, 2, 3, 4), dtype=np.float32)
    cache.append(0, k, v)
    cache.advance(3)
    try:
        cache.append(0, k, v)
        cache.advance(3)
        raise AssertionError("Должно было упасть")
    except ValueError as e:
        assert "переполнен" in str(e)


check("KVCache: overflow protection", test_kvcache_overflow)


def test_kvcache_reset():
    cache = nw.KVCache(num_layers=1, num_kv_heads=2, head_dim=4, max_seq_len=10)
    cache.append(0, np.zeros((1, 2, 1, 4)), np.zeros((1, 2, 1, 4)))
    cache.advance(1)
    assert cache.cur_len == 1
    cache.reset()
    assert cache.cur_len == 0


check("KVCache: reset", test_kvcache_reset)


# ============================================================
# 2. RoPECache
# ============================================================
print("\n[2] RoPECache")


def test_rope_shapes():
    cache = nw.RoPECache(head_dim=16, max_seq_len=10)
    cos, sin = cache.get(5)
    assert cos.shape == (5, 16)
    assert sin.shape == (5, 16)
    assert np.allclose(cos[0], 1.0)
    assert np.allclose(sin[0], 0.0)


check("RoPECache: shapes + position 0", test_rope_shapes)


def test_rope_apply():
    cache = nw.RoPECache(head_dim=16, max_seq_len=10)
    cos, sin = cache.get(3)
    q = np.random.randn(1, 4, 3, 16).astype(np.float32)
    k = np.random.randn(1, 2, 3, 16).astype(np.float32)
    q_rot, k_rot = nw.RoPECache.apply(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not np.isnan(q_rot).any()


check("RoPECache: apply", test_rope_apply)


def test_rope_scaling():
    cache_linear = nw.RoPECache(head_dim=16, max_seq_len=10,
                                rope_scaling={"type": "linear", "factor": 2.0})
    cache_llama3 = nw.RoPECache(head_dim=16, max_seq_len=10,
                                rope_scaling={"type": "llama3", "factor": 8.0})
    assert cache_linear.cos.shape == (10, 16)
    assert cache_llama3.cos.shape == (10, 16)


check("RoPECache: linear + llama3 scaling", test_rope_scaling)


# ============================================================
# 3. FusedLinear + FusedMLP
# ============================================================
print("\n[3] FusedLinear / FusedMLP")


def test_fused_qkv():
    W_q = np.random.randn(128, 128).astype(np.float32)
    W_k = np.random.randn(64, 128).astype(np.float32)
    W_v = np.random.randn(64, 128).astype(np.float32)
    fuse = nw.fuse_qkv(W_q, W_k, W_v)
    x = np.random.randn(4, 128).astype(np.float32)
    out = fuse(x)
    assert out["q"].shape == (4, 128)
    assert out["k"].shape == (4, 64)
    assert out["v"].shape == (4, 64)
    assert np.allclose(out["q"], x @ W_q.T)
    assert np.allclose(out["k"], x @ W_k.T)


check("FusedLinear: fuse_qkv", test_fused_qkv)


def test_fused_mlp():
    W_gate = np.random.randn(256, 128).astype(np.float32)
    W_up = np.random.randn(256, 128).astype(np.float32)
    W_down = np.random.randn(128, 256).astype(np.float32)
    mlp = nw.FusedMLP(W_gate, W_up, W_down, activation="silu")
    x = np.random.randn(4, 128).astype(np.float32)
    y = mlp(x)
    assert y.shape == (4, 128)

    # Проверка эквивалентности с ручным расчётом
    gate = x @ W_gate.T
    up = x @ W_up.T
    silu_gate = gate / (1.0 + np.exp(-gate))
    hidden = silu_gate * up
    expected = hidden @ W_down.T
    assert np.allclose(y, expected, atol=1e-5)


check("FusedMLP: silu equivalence", test_fused_mlp)


def test_fused_mlp_activations():
    W_gate = np.random.randn(64, 32).astype(np.float32)
    W_up = np.random.randn(64, 32).astype(np.float32)
    W_down = np.random.randn(32, 64).astype(np.float32)
    x = np.random.randn(2, 32).astype(np.float32)
    for act in ("silu", "gelu", "relu", "identity"):
        mlp = nw.FusedMLP(W_gate, W_up, W_down, activation=act)
        y = mlp(x)
        assert y.shape == (2, 32)
        assert not np.isnan(y).any()


check("FusedMLP: 4 активации", test_fused_mlp_activations)


# ============================================================
# 4. Fast softmax
# ============================================================
print("\n[4] fast_softmax")


def test_softmax_numerical_stability():
    x = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    y = nw.fast_softmax(x)
    assert not np.isnan(y).any()
    assert not np.isinf(y).any()
    assert abs(y.sum() - 1.0) < 1e-6


check("fast_softmax: numerical stability", test_softmax_numerical_stability)


def test_log_softmax():
    x = np.array([1.0, 2.0, 3.0])
    log_y = nw.log_softmax(x)
    y = np.exp(log_y)
    assert np.allclose(y, nw.fast_softmax(x), atol=1e-6)


check("log_softmax: inverse of exp", test_log_softmax)


# ============================================================
# 5. Attention
# ============================================================
print("\n[5] scaled_dot_product_attention")


def test_mha():
    q = np.random.randn(1, 4, 5, 16).astype(np.float32)
    k = np.random.randn(1, 4, 5, 16).astype(np.float32)
    v = np.random.randn(1, 4, 5, 16).astype(np.float32)
    out = nw.scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 4, 5, 16)
    assert not np.isnan(out).any()


check("attention: MHA", test_mha)


def test_gqa_broadcast_equivalence():
    np.random.seed(42)
    q = np.random.randn(1, 8, 5, 16).astype(np.float32)
    k = np.random.randn(1, 2, 5, 16).astype(np.float32)
    v = np.random.randn(1, 2, 5, 16).astype(np.float32)
    out_broadcast = nw.scaled_dot_product_attention(q, k, v)

    repeat = 4
    k_rep = np.repeat(k, repeat, axis=1)
    v_rep = np.repeat(v, repeat, axis=1)
    scores = np.matmul(q, np.swapaxes(k_rep, -1, -2)) / np.sqrt(16)
    weights = nw.fast_softmax(scores, axis=-1)
    out_repeat = np.matmul(weights, v_rep)

    diff = np.abs(out_broadcast - out_repeat).max()
    assert diff < 1e-5, f"diff={diff}"


check("attention: GQA broadcast == repeat", test_gqa_broadcast_equivalence)


def test_mqa():
    q = np.random.randn(1, 8, 5, 16).astype(np.float32)
    k = np.random.randn(1, 1, 5, 16).astype(np.float32)
    v = np.random.randn(1, 1, 5, 16).astype(np.float32)
    out = nw.scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 8, 5, 16)
    assert not np.isnan(out).any()


check("attention: MQA (num_kv_heads=1)", test_mqa)


def test_causal_mask():
    scores = np.zeros((3, 3))
    masked = nw.apply_causal_mask(scores, 3)
    # Нижний треугольник + диагональ = 0, верхний = -inf
    assert masked[0, 1] == -np.inf
    assert masked[1, 0] == 0.0
    assert masked[2, 2] == 0.0


check("apply_causal_mask", test_causal_mask)


# ============================================================
# 6. Top-k / Top-p
# ============================================================
print("\n[6] top_k / top_p")


def test_top_k():
    logits = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    filtered = nw.top_k_filter(logits, k=2)
    assert filtered[1] == 5.0
    assert filtered[3] == 4.0
    assert filtered[0] == -np.inf
    assert filtered[2] == -np.inf


check("top_k_filter", test_top_k)


def test_top_p():
    logits = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    filtered = nw.top_p_filter(logits, p=0.9)
    assert not np.isnan(filtered).any()
    # Самый вероятный точно остался
    assert filtered[1] == 5.0


check("top_p_filter", test_top_p)


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print(f"  PASSED: {passed} / {passed + failed}")
print(f"  FAILED: {failed}")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)