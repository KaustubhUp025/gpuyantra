---
paths:
  - "kernelsmith/verifier/**"
  - "tests/test_correctness*"
  - "tests/test_static_checker*"
  - "tests/test_sandbox*"
  - "tests/test_reward*"
---

# Verifier Rules

The verifier is the trust anchor. Read Section 5 of the kernelsmith-spec skill before editing.

- 5 seeds × 3 shapes, atol=rtol=1e-2. All 15 checks must pass.
- do_bench warmup=150, rep=200, return_mode="median"
- Static AST checker must cover all 7 reward-hack patterns (see spec §5.3)
- Every candidate runs in subprocess with SIGKILL timeout
- GPU health probe after every candidate
- chmod 444 on verifier scripts
- TF32 baseline always enabled: torch.set_float32_matmul_precision('high')
- Milestone reward: -1 (fail) / +1 (correct, not faster) / +2 (beats eager >5%) / +3 (beats both >5%)
