#!/usr/bin/env python
"""Seed Firestore with one hand-written, known-good RMSNorm skill.

This is the cold-start seed for bottleneck-indexed retrieval: without at least one
row in `skills`, the very first run has nothing to retrieve. The embedding is a REAL
gemini-embedding-001 @768 vector over the bottleneck fingerprint text (never a
random or zero vector — retrieval quality depends on it).

Usage:
    python scripts/seed_skill.py            # embed + write to Firestore
    python scripts/seed_skill.py --dry-run  # embed + print, no write
"""

from __future__ import annotations

import argparse
import json

from kernelsmith.memory.embeddings import embed_768
from kernelsmith.memory.firestore_store import upsert_skill
from kernelsmith.memory.schemas import BottleneckFingerprint, SkillRecord

SKILL_ID = "rmsnorm_fp16_l4_v1"

#: Warm start from the replay buffer: 3 hand-tested pulls, all +3 -> mean reward 3.0.
SEED_BANDIT_PULLS = 3
SEED_BANDIT_TOTAL_REWARD = 9.0

# Hand-written fused RMSNorm. Numerically matches Qwen2RMSNorm:
#   y = x * rsqrt(mean(x^2) + eps) * weight, with the reduction in fp32.
RMSNORM_KERNEL_SOURCE = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(
    X, W, Y,
    stride_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. Two-pass over the row so N may exceed BLOCK_SIZE."""
    row = tl.program_id(0)
    X += row * stride_row
    Y += row * stride_row

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    # Reduce in fp32: matching eager requires mean(x^2) + eps BEFORE the rsqrt.
    inv_rms = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, (x * inv_rms * w).to(Y.dtype.element_ty), mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in for Qwen2RMSNorm.forward. Accepts [..., H], returns the same shape."""
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x2d.shape
    y = torch.empty_like(x2d)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)
    num_warps = 8 if BLOCK_SIZE >= 1024 else 4

    _rmsnorm_fwd[(M,)](
        x2d, weight.contiguous(), y,
        x2d.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.reshape(orig_shape)
'''.strip()

FINGERPRINT = BottleneckFingerprint(
    op_family="norm",
    hardware="L4",
    memory_throughput_gbps=212.4,  # ~71% of the L4's 300.1 GB/s peak
    achieved_occupancy=0.62,
    arithmetic_intensity=0.5,  # ~2 flops per byte moved: squarely memory-bound
    is_memory_bound=True,
    is_compute_bound=False,
    tile_size_hint=1024,
)


def build_seed_skill() -> SkillRecord:
    """Build the seed SkillRecord with a real 768-dim, L2-normalized embedding."""
    embedding = embed_768(FINGERPRINT.to_embedding_text())
    return SkillRecord(
        skill_id=SKILL_ID,
        op_signature="rmsnorm_fp16_[B,S,H]",
        op_family="norm",
        hardware="L4",
        bottleneck_fingerprint=FINGERPRINT,
        winning_kernel_source=RMSNORM_KERNEL_SOURCE,
        speedup_vs_eager=1.71,
        speedup_vs_torch_compile=1.06,
        fix_rule=(
            "Memory-bound row-wise reduction: fuse the square-sum, rsqrt, and weight "
            "multiply into a single kernel so the row is read twice instead of the four "
            "round-trips eager PyTorch makes. Reduce in fp32, store in the input dtype. "
            "One program per row, BLOCK_SIZE = min(next_pow2(H), 1024), num_warps=8 at 1024."
        ),
        embedding=embedding,
        tags=["rmsnorm", "norm", "memory-bound", "fused", "seed", "hand-written"],
        # Bandit warm start (spec 9): three hand-tested runs at reward +3, so mean = 3.0.
        # Without this the seed arm looks unpulled and UCB1 explores it exactly as it
        # would a kernel nobody has ever run — throwing away results we already have.
        bandit_pulls=SEED_BANDIT_PULLS,
        bandit_total_reward=SEED_BANDIT_TOTAL_REWARD,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Embed and print the record, but do not write to Firestore.",
    )
    args = parser.parse_args()

    skill = build_seed_skill()
    assert len(skill.embedding) == 768, f"Expected 768 dims, got {len(skill.embedding)}"

    if args.dry_run:
        preview = skill.model_dump()
        preview["embedding"] = f"<{len(skill.embedding)} floats, first 4: {skill.embedding[:4]}>"
        preview["winning_kernel_source"] = f"<{len(skill.winning_kernel_source)} chars>"
        print(json.dumps(preview, indent=2, default=str))
        print("\nDRY RUN — nothing written to Firestore.")
        return

    result = upsert_skill(skill)
    print(f"{result}: skills/{skill.skill_id} (fingerprint: {FINGERPRINT.to_embedding_text()})")
    print(
        f"bandit warm start: {skill.bandit_pulls} pulls, "
        f"mean reward {skill.bandit_total_reward / skill.bandit_pulls:.1f}"
    )


if __name__ == "__main__":
    main()
