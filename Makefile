.PHONY: setup demo test test-unit test-int lint format seed-skill create-index \
        serve-inference serve-ui harden unharden check-harden export-firestore

# --- One-command setup (README "Quick start") -------------------------------
# `uv sync --frozen` refuses to re-resolve: a fresh L4 gets the exact transitive
# closure recorded in the committed uv.lock, not merely deps satisfying pyproject.
# create-index takes several minutes to build and is idempotent-by-failure (a
# second run reports ALREADY_EXISTS), so it is tolerated rather than fatal.
setup:
	@echo "=== 1/4 installing pinned dependencies ==="
	uv sync --frozen
	@echo "=== 2/4 hardening verifier scripts (chmod 444) ==="
	@$(MAKE) --no-print-directory harden
	@echo "=== 3/4 creating the Firestore composite vector index (minutes; ok if it exists) ==="
	-bash infra/create_index.sh
	@echo "=== 4/4 seeding the RMSNorm skill ==="
	uv run python scripts/seed_skill.py
	@echo "=== setup complete — run 'make demo' ==="

# DEMO_ARGS lets a box that cannot host the 3 GB server run the agent half alone:
#   make demo DEMO_ARGS=--no-server
DEMO_ARGS ?=

# CUBLAS_WORKSPACE_CONFIG is exported here as well as by seed_everything(): cuBLAS
# reads it when its handle is created, and belt-and-braces costs nothing.
demo:
	@echo "=== KernelSmith Demo ==="
	CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m kernelsmith.run_demo $(DEMO_ARGS)

test-unit:
	uv run pytest tests/ -k "not integration and not chaos" -v --tb=short

test-int:
	uv run pytest tests/ -k "integration" -v --tb=short

test: test-unit test-int

lint:
	uv run ruff check kernelsmith/ tests/ scripts/
	uv run ruff format --check kernelsmith/ tests/ scripts/

format:
	uv run ruff format kernelsmith/ tests/ scripts/

seed-skill:
	uv run python scripts/seed_skill.py

create-index:
	bash infra/create_index.sh

serve-inference:
	uv run uvicorn kernelsmith.inference_server.server:app --host 0.0.0.0 --port 8000

serve-ui:
	uv run streamlit run kernelsmith/ui/streamlit_app.py --server.port 8501

# --- Security (spec 12) -----------------------------------------------------
# The verifier is the trust anchor: it is the only thing standing between a
# reward-hacking kernel and a headline number. Generated code runs as a
# subprocess owned by this same uid, so nothing at the OS level stops it from
# rewriting the checker that is about to judge it — except the write bit.
# 444 is not a sandbox; it is the cheap interlock that makes tampering
# deliberate rather than accidental.
#
# Git tracks only the executable bit, so these modes do not survive a fresh
# clone. `make setup` re-applies them, and `make check-harden` verifies.
VERIFIER_SCRIPTS := \
	kernelsmith/verifier/correctness.py \
	kernelsmith/verifier/timing.py \
	kernelsmith/verifier/static_checker.py \
	kernelsmith/verifier/reward.py

harden:
	chmod 444 $(VERIFIER_SCRIPTS)
	@ls -l $(VERIFIER_SCRIPTS)

# Editing a verifier file requires taking the write bit back deliberately.
# Re-run `make harden` immediately afterwards.
unharden:
	chmod 644 $(VERIFIER_SCRIPTS)
	@ls -l $(VERIFIER_SCRIPTS)

check-harden:
	@fail=0; for f in $(VERIFIER_SCRIPTS); do \
		mode=$$(stat -c '%a' $$f); \
		if [ "$$mode" != "444" ]; then echo "NOT HARDENED: $$f is $$mode, expected 444"; fail=1; \
		else echo "ok $$f 444"; fi; \
	done; \
	if [ $$fail -ne 0 ]; then echo "run 'make harden'"; exit 1; fi

# --- Reproducibility (spec 11) ----------------------------------------------
# Snapshot the skill library + bandit state so a demo can be replayed against
# the same memory it was recorded with.
export-firestore:
	bash scripts/export_firestore.sh
