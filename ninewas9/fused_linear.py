"""FusedLinear — объединение нескольких линейных слоёв в один matmul."""
import numpy as np


class FusedLinear:
    """Объединяет несколько матриц весов в одну и делает один matmul.

    Параметры:
        weights: список кортежей (name, W)
                 где W — 2D массив формы (out_features, in_features)
    """

    def __init__(self, weights):
        if not weights:
            raise ValueError("FusedLinear: пустой список весов")

        self.names = []
        self.shapes = []
        matrices = []

        for name, W in weights:
            if W.ndim != 2:
                raise ValueError(f"FusedLinear: {name}.ndim={W.ndim}, ожидалось 2")
            self.names.append(name)
            self.shapes.append(W.shape)
            matrices.append(W)

        self.fused_weight = np.concatenate(matrices, axis=0)
        self.fused_weight_T = self.fused_weight.T
        self.total_out = self.fused_weight.shape[0]

    def __call__(self, x):
        if x.shape[-1] != self.shapes[0][1]:
            raise ValueError(
                f"FusedLinear: ожидалось in_features={self.shapes[0][1]}, "
                f"получено {x.shape[-1]}"
            )
        out = x @ self.fused_weight_T
        result = {}
        offset = 0
        for name, shape in zip(self.names, self.shapes):
            n = shape[0]
            result[name] = out[..., offset:offset + n]
            offset += n
        return result

    def __repr__(self):
        parts = [f"{name}{shape}" for name, shape in zip(self.names, self.shapes)]
        return f"<FusedLinear [{', '.join(parts)}] fused={self.fused_weight.shape}>"


def fuse_qkv(w_q, w_k, w_v):
    return FusedLinear([("q", w_q), ("k", w_k), ("v", w_v)])


def fuse_gate_up(w_gate, w_up):
    return FusedLinear([("gate", w_gate), ("up", w_up)])


class FusedMLP:
    """Объединённый MLP: [gate|up] @ x, потом down @ (act(gate) * up)."""

    def __init__(self, w_gate, w_up, w_down, activation="silu"):
        self.fused_gate_up = fuse_gate_up(w_gate, w_up)
        self.w_down = w_down
        self.w_down_T = w_down.T
        self.activation = activation

    def _apply_activation(self, x):
        if callable(self.activation):
            return self.activation(x)
        if self.activation in ("silu", "swish"):
            return x / (1.0 + np.exp(-x))
        if self.activation == "gelu":
            return 0.5 * x * (1.0 + np.tanh(
                0.7978845608 * (x + 0.044715 * x ** 3)
            ))
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "identity":
            return x
        raise ValueError(f"FusedMLP: неизвестная активация '{self.activation}'")

    def __call__(self, x):
        gate_up = self.fused_gate_up(x)
        gate = gate_up["gate"]
        up = gate_up["up"]
        hidden = self._apply_activation(gate) * up
        return hidden @ self.w_down_T


__all__ = [
    "FusedLinear", "FusedMLP",
    "fuse_qkv", "fuse_gate_up",
]