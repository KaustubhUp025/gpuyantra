.PHONY: demo test test-unit test-int lint format seed-skill create-index serve-inference serve-ui

demo:
	@echo "=== KernelSmith Demo ==="
	@echo "Seeding reproducibility..."
	CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -c "import kernelsmith; kernelsmith.run_demo()"

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
