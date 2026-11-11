# The same commands CI runs: if `make check smoke` passes locally, CI should pass too.
#
# `uv run --locked` fails if uv.lock is out of date with pyproject.toml, instead of quietly
# re-resolving, so a stale lockfile is caught here as well as in CI.

IMAGE ?= ssv2:local
# Pinned by digest, like every other image. Keep the tag in the comment in sync.
TRIVY_IMAGE ?= aquasec/trivy:0.76.1@sha256:790fd66d1697395d8693d1acaa1ea702406a20f109b6fc7bac8db623eca6c06c
TRIVY_CACHE ?= $(HOME)/.cache/trivy
# Extra flags for unusual networks, e.g. behind a TLS-intercepting proxy:
#   make smoke DOCKER_BUILD_FLAGS="--secret id=extra_ca,src=/path/ca.crt"
DOCKER_BUILD_FLAGS ?=
DOCKER_RUN_FLAGS ?=

UV_RUN := uv run --locked
# Lint needs only ruff, so it skips PyTorch and the rest of the project's dependencies.
UV_RUN_LINT := uv run --locked --only-group lint

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check run docker-build smoke-test smoke scan clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

install: ## Create .venv from uv.lock and install the git pre-commit hook
	uv sync --locked
	$(UV_RUN) pre-commit install

lint: ## ruff check + ruff format --check
	$(UV_RUN_LINT) ruff check .
	$(UV_RUN_LINT) ruff format --check .

format: ## Fix lint and formatting in place
	$(UV_RUN_LINT) ruff format .
	$(UV_RUN_LINT) ruff check --fix .

typecheck: ## mypy --strict on src and tests
	$(UV_RUN) mypy

test: ## Unit tests only (fast: no model download)
	$(UV_RUN) pytest -m "not integration"

check: lint typecheck test ## lint + typecheck + test

run: ## API with auto-reload on http://localhost:8000 (docs at /docs)
	$(UV_RUN) uvicorn ssv2.api.app:create_app --factory --reload

docker-build: ## Build the image as $(IMAGE)
	docker build $(DOCKER_BUILD_FLAGS) -t $(IMAGE) .

smoke-test: ## Run and query an already-built $(IMAGE)
	./scripts/smoke_test.sh $(IMAGE)

smoke: docker-build smoke-test ## Build the image, run it, query it

scan: ## Scan $(IMAGE) for fixable HIGH/CRITICAL vulnerabilities with Trivy
	@mkdir -p $(TRIVY_CACHE)
	docker save $(IMAGE) -o .image.tar
	docker run --rm $(DOCKER_RUN_FLAGS) -v "$(CURDIR)/.image.tar:/image.tar:ro" -v "$(TRIVY_CACHE):/root/.cache/trivy" \
		$(TRIVY_IMAGE) image --input /image.tar \
		--scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 --no-progress; \
		status=$$?; rm -f .image.tar; exit $$status

clean: ## Remove caches and build leftovers
	rm -rf .pytest_cache .ruff_cache .mypy_cache .image.tar
