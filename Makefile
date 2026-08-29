.PHONY: setup demo audit audit-all test test-unit test-int lint format seed-skill \
        create-index serve-inference serve-ui serve-demo demo-with-dashboard \
        harden unharden check-harden export-firestore

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

# --- Audit (spec 7) ---------------------------------------------------------
# On CPU the audit builds every module from config.json on the meta device — no GPU,
# no weight download, ~4s for any of the three registered models. `audit` follows
# the GPU when there is one (spec default), and CUDA mode DOES load real weights so
# do_bench has something to time; on a cold cache that is a multi-GB download.
# `audit-all` defaults to CPU regardless — see run_audit_all.
# AUDIT_ARGS reaches the subcommand:
#   make audit AUDIT_ARGS="--device cpu"
#   make audit AUDIT_ARGS="--model gpt2 --device cuda"
AUDIT_ARGS ?=

audit:
	GOOGLE_CLOUD_PROJECT=gpuyantra uv run python -m kernelsmith.run_demo audit $(AUDIT_ARGS)

# The sweep runs on CPU unless --device says otherwise (see run_audit_all): three
# models on CUDA means ~3.4 GB of weights and minutes of do_bench, for a bandwidth
# column the comparison table does not have.
audit-all:
	GOOGLE_CLOUD_PROJECT=gpuyantra uv run python -m kernelsmith.run_demo audit --all $(AUDIT_ARGS)

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

# --- Demo dashboard (Task 12) -----------------------------------------------
# A SECOND Streamlit app, on a different port, for recording the video. The
# operator dashboard on 8501 is untouched (CLAUDE.md rule 14) and the two run
# side by side. `--theme.base dark` is not decoration: st.set_page_config has no
# theme parameter, so this flag is the only thing that pins the dark theme for
# someone whose Streamlit config defaults to light.
serve-demo:
	GOOGLE_CLOUD_PROJECT=gpuyantra \
	  uv run streamlit run kernelsmith/ui/demo_dashboard.py \
	  --server.port 8502 --theme.base dark

# One command for the recording take: inference server, demo dashboard, then the
# agent run whose events both the dashboard and data/traces/ pick up.
#
# The two servers are backgrounded from this recipe, so they are NOT children of
# your shell and `make demo-with-dashboard` will not clean them up on Ctrl-C.
# Stop them afterwards with:  pkill -f 'uvicorn kernelsmith' ; pkill -f demo_dashboard
#
# Run this on the VM: it wants the L4 for both the served model and the verifier.
demo-with-dashboard:
	@echo "=== 1/3 inference server on :8000 ==="
	$(MAKE) serve-inference &
	sleep 5
	@echo "=== 2/3 demo dashboard on :8502 ==="
	$(MAKE) serve-demo &
	sleep 3
	@echo "=== 3/3 agent run ==="
	$(MAKE) demo
	@echo "Demo complete. Dashboard at http://localhost:8502"
	@echo "Trace saved to data/traces/"

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
