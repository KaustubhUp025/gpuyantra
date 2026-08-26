"""Data models for the entire system. Every structured boundary uses these."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from kernelsmith.config import EMBEDDING_DIM


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (datetime.utcnow() is deprecated in 3.12+)."""
    return datetime.now(UTC)


class BottleneckFingerprint(BaseModel):
    """Roofline-derived fingerprint of WHY an op is slow."""

    op_family: str = Field(..., description="norm | rope | mlp | elementwise | reduction")
    hardware: str = Field(default="L4")
    memory_throughput_gbps: float
    achieved_occupancy: float = Field(ge=0.0, le=1.0)
    arithmetic_intensity: float = Field(ge=0.0)
    is_memory_bound: bool
    is_compute_bound: bool
    tile_size_hint: int = Field(ge=1)

    def to_embedding_text(self) -> str:
        """Convert to text for embedding. This IS the bottleneck-indexed retrieval key."""
        return (
            f"op={self.op_family} "
            f"mem_bound={self.is_memory_bound} "
            f"ai={self.arithmetic_intensity:.1f} "
            f"tile={self.tile_size_hint} "
            f"hw={self.hardware}"
        )


class KernelDraft(BaseModel):
    """Output of the Coder agent. One Triton kernel attempt."""

    code: str = Field(..., description="Complete Python source with @triton.jit kernel + wrapper")
    entrypoint: str = Field(..., description="Name of the callable wrapper function")
    block_sizes: dict = Field(
        default_factory=dict, description="BLOCK_SIZE and num_warps suggestions"
    )
    rationale: str = Field(
        ..., description="Why this kernel should be faster, referencing the fingerprint"
    )
    adapter_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Deployment contract: kernel parameter name -> attribute on the module "
            'being patched, e.g. {"weight": "weight", "eps": "variance_epsilon"}. The '
            "forward's input tensor is implicit and must NOT appear here. Validated by "
            "verifier.adapter_mapping before the kernel is run; empty means fall back "
            "to the hard-coded per-op adapter."
        ),
    )


class Verdict(BaseModel):
    """Output of the Judge agent after calling verifier_tool."""

    reward: int = Field(..., ge=-1, le=3)
    correctness_pass: bool
    speedup_vs_eager: float
    speedup_vs_compile: float
    next_action: str = Field(..., description="STOP or a concrete fix instruction for the Coder")
    stop: bool
    stderr_tail: str = Field(default="", description="Last 500 chars of subprocess stderr if any")
    latency_ms_by_shape: dict = Field(default_factory=dict)


class SkillRecord(BaseModel):
    """A verified, reusable kernel skill stored in Firestore."""

    skill_id: str
    op_signature: str  # e.g., "rmsnorm_fp16_[B,S,H]"
    op_family: str  # "norm" | "rope" | "mlp" | "elementwise" | "reduction"
    hardware: str = "L4"
    bottleneck_fingerprint: BottleneckFingerprint
    winning_kernel_source: str  # Complete Python source
    speedup_vs_eager: float
    speedup_vs_torch_compile: float
    fix_rule: str  # Human-readable description of the optimization applied
    embedding: list[float]  # 768-dim, L2-normalized
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    # Bandit stats
    bandit_pulls: int = 0
    bandit_total_reward: float = 0.0

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dim(cls, v: list[float]) -> list[float]:
        assert len(v) == EMBEDDING_DIM, f"Embedding must be {EMBEDDING_DIM}-dim, got {len(v)}"
        return v


class RunRecord(BaseModel):
    """Metadata for one complete optimization run."""

    run_id: str
    task_ref: str
    started_at: datetime
    ended_at: datetime | None = None
    final_reward: int = -1
    total_iterations: int = 0
    total_tokens_spent: int = 0
    cost_estimate_usd: float = 0.0
    status: str = "running"  # running | completed | failed


class TraceRecord(BaseModel):
    """One iteration's trace within a run (Firestore subcollection: runs/{run_id}/traces)."""

    iteration: int
    agent: str
    prompt_summary: str
    response_summary: str
    reward: int
    latency_ms_by_shape: dict
    timestamp: datetime = Field(default_factory=_utcnow)
