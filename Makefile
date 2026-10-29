# =============================================================================
# WAFer - Gemma-Clef Containerized Decision Engine Makefile
# =============================================================================

# Container runtime detection: prefers podman, falls back to docker
RUNTIME ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
ifeq ($(RUNTIME),)
$(error "Neither podman nor docker was found in PATH.")
endif
RUNTIME_NAME := $(notdir $(RUNTIME))

IMAGE_TAG ?= wafer-engine:dev
WORKSPACE ?= $(CURDIR)
LOG_FILE ?= data/sample_traffic.log
PYTEST_ARGS ?= -v
CLASSIFY_ARGS ?=

# Extract HF_TOKEN if available in environment or .env
ifneq (,$(wildcard .env))
HF_TOKEN ?= $(shell grep -E '^ *(export +)?HF_TOKEN=' .env 2>/dev/null | sed -E "s/^ *(export +)?HF_TOKEN=[\"']?([^\"']*)[\"']?/\2/")
endif
ENV_FLAGS := $(if $(HF_TOKEN),-e HF_TOKEN=$(HF_TOKEN),)

# GPU auto-detection or override (GPU=0 to disable passthrough, GPU=1 to force)
HAS_NVIDIA := $(shell command -v nvidia-smi 2>/dev/null)
ifeq ($(GPU),0)
GPU_FLAGS :=
else ifeq ($(GPU),1)
GPU_FLAGS := --gpus all
else
GPU_FLAGS := $(if $(HAS_NVIDIA),--gpus all,)
endif

# Common container invocation flags
RUN_FLAGS := --rm -e PYTHONPATH=/app $(ENV_FLAGS) $(GPU_FLAGS) -v $(WORKSPACE):/app:Z -w /app

.PHONY: all help build test test-unit test-integration test-fast inspect inspect-embeddinggemma train-embeddinggemma classify lock check-lock clean shell

all: help

help:
	@echo "WAFer - Gemma-Clef Containerized Decision Engine"
	@echo ""
	@echo "Usage: make <target> [ARGS='...']"
	@echo ""
	@echo "Targets:"
	@echo "  build                  Build the container image ($(IMAGE_TAG)) via $(RUNTIME_NAME)"
	@echo "  test                   Run the full pytest suite in container (supports ARGS='...')"
	@echo "  test-unit              Run unit tests only"
	@echo "  test-integration       Run integration tests only"
	@echo "  test-fast              Run unit tests (skips long model training)"
	@echo "  inspect                Evaluate sample traffic logs (default: $(LOG_FILE))"
	@echo "  inspect-embeddinggemma Evaluate sample traffic using EmbeddingGemma 2"
	@echo "  train-embeddinggemma   Fine-tune EmbeddingGemma 2 + LoRA on WAF dataset"
	@echo "  classify               Run log classifier and rules test summary"
	@echo "  lock                   Generate / update uv.lock from pyproject.toml"
	@echo "  check-lock             Verify uv.lock is in sync with pyproject.toml"
	@echo "  shell                  Open an interactive shell inside the container"
	@echo "  clean                  Remove cache directories and temporary checkpoints"
	@echo "  help                   Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make test"
	@echo "  make test ARGS='-k test_rules'"
	@echo "  make inspect LOG_FILE=path/to/logfile.log"

build:
	@echo "==> Building container image '$(IMAGE_TAG)' with $(RUNTIME_NAME)..."
	$(RUNTIME) build -t $(IMAGE_TAG) -f docker/Dockerfile.engine .

test:
	@echo "==> Running full pytest suite in container..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) pytest tests/ $(PYTEST_ARGS) $(ARGS)

test-unit:
	@echo "==> Running unit tests in container..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) pytest tests/unit/ $(PYTEST_ARGS) $(ARGS)

test-integration:
	@echo "==> Running integration tests in container..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) pytest tests/integration/ $(PYTEST_ARGS) $(ARGS)

test-fast:
	@echo "==> Running fast tests (unit tests)..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) pytest tests/unit/ $(PYTEST_ARGS) $(ARGS)

inspect:
	@echo "==> Running log inspection on $(LOG_FILE)..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) python3 wafer/eval/inspect_logs.py $(LOG_FILE)

inspect-embeddinggemma:
	@echo "==> Running log inspection with EmbeddingGemma 2..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) python3 -m wafer.eval.inspect_logs $(LOG_FILE) --model-id embeddinggemma-2 $(ARGS)

train-embeddinggemma:
	@echo "==> Fine-tuning EmbeddingGemma 2 on WAF traffic..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) python3 -m wafer.eval.inspect_logs $(LOG_FILE) --model-id embeddinggemma-2 --train --epochs 3 $(ARGS)

classify:
	@echo "==> Running log classifier in container..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) python3 -m wafer.data.log_classifier --test-rules --summary $(CLASSIFY_ARGS)

lock:
	@echo "==> Updating uv.lock in container..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) uv lock

check-lock:
	@echo "==> Verifying uv.lock consistency..."
	$(RUNTIME) run $(RUN_FLAGS) $(IMAGE_TAG) uv lock --check

shell:
	@echo "==> Spawning interactive shell in $(IMAGE_TAG)..."
	$(RUNTIME) run -it $(RUN_FLAGS) $(IMAGE_TAG) bash

clean:
	@echo "==> Cleaning cache files and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov wafer.egg-info
	@echo "==> Clean complete."
