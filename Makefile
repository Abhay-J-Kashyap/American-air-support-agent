.PHONY: help setup lint fmt typecheck test check-providers data intents index golden eval eval-live freeze-cache report clean

PY := python
CONFIG ?= config/config.yaml

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and dev dependencies
	$(PY) -m pip install -e ".[dev]"

lint:  ## ruff check
	ruff check src tests

fmt:  ## ruff format
	ruff format src tests

typecheck:  ## mypy
	mypy

test:  ## Run the test suite (no network)
	pytest -m "not live"

check-providers:  ## Ping every model and print the limits the API reports
	$(PY) -m aa_agent.cli check-providers --config $(CONFIG)

# ---- pipeline -----------------------------------------------------------
data:  ## Stream-filter the raw CSV down to the brand slice, then drop the raw file
	$(PY) -m aa_agent.cli ingest --config $(CONFIG)

intents:  ## Cluster inbound messages and propose an intent taxonomy
	$(PY) -m aa_agent.cli discover-intents --config $(CONFIG)

index:  ## Build the retrieval index over the training window only
	$(PY) -m aa_agent.cli build-index --config $(CONFIG)

golden:  ## Launch the labelling CLI for the golden set
	$(PY) -m aa_agent.cli label --config $(CONFIG)

eval:  ## HEADLINE RESULTS from cache. No API key, no spend, <15 min.
	$(PY) -m aa_agent.cli evaluate --config $(CONFIG) --offline

eval-live:  ## Regenerate every prediction against the live APIs (hours)
	$(PY) -m aa_agent.cli evaluate --config $(CONFIG)

report:  ## Render metrics, tables and figures into report/
	$(PY) -m aa_agent.cli report --config $(CONFIG)

freeze-cache:  ## Compress the response cache for committing
	gzip -9 -kf artifacts/llm_cache.jsonl && ls -lh artifacts/llm_cache.jsonl.gz

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
