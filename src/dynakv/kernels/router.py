"""Runtime routing between the short and long compressed attention kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelPolicy:
    """Select the fused kernel below ``switch_tokens`` and two-stage above it."""

    switch_tokens: int = 500

    def select(self, cache_length: int) -> str:
        if cache_length < 0:
            raise ValueError("cache_length must be non-negative")
        return "short_fused" if cache_length < self.switch_tokens else "long_two_stage"


def compressed_decode_attention(
    *, q, k_flat, k_up, v_flat, v_up, key_lengths, key_offsets,
    value_lengths, value_offsets, theta: float, cache_length: int,
    policy: KernelPolicy | None = None,
):
    """Run the kernel selected from the current physical cache length.

    Imports are deliberately lazy: installing the research/training package does
    not require Triton, while the optimized path still fails clearly on a CPU or
    unsupported platform.
    """
    policy = policy or KernelPolicy()
    selected = policy.select(cache_length)
    if selected == "short_fused":
        from .fused import fused_attention_dispatch

        return fused_attention_dispatch(
            q=q,
            k_flat=k_flat,
            k_up=k_up,
            v_flat=v_flat,
            v_up=v_up,
            l_k=key_lengths,
            o_k=key_offsets,
            l_v=value_lengths,
            o_v=value_offsets,
            theta=theta,
        )

    from .two_stage import abx_sparse, avx

    scores = abx_sparse(
        q.squeeze(2).contiguous(),
        k_up,
        k_flat,
        key_lengths,
        offsets=key_offsets,
        theta=theta,
    )
    return avx(scores, v_flat, v_up, value_lengths, value_offsets)
