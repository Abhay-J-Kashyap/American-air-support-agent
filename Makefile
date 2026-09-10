.PHONY: help setup setup-dev lock lint fmt typecheck test check-providers data intents taxonomy index golden eval eval-live freeze-cache report clean

PY := python
CONFIG ?= config/config.yaml

help:  ## List available targets
	@$(PY) scripts/help.py

setup:  ## Install from the lock file (reproducible)
	@$(PY) scripts/require_lock.py
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install -e . --no-deps

setup-dev:  ## Resolve fresh from pyproject (use when adding/upgrading deps)
	$(PY) -m pip install -e ".[dev]"

lock:  ## Regenerate the lock file after changing dependencies
	$(PY) -m pip freeze --exclude-editable > requirements.lock

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

intents:  ## Cluster first-turn customer messages and propose an intent taxonomy
	$(PY) -m aa_agent.cli discover-intents --config $(CONFIG)

taxonomy:  ## Validate and display the curated taxonomy (config/taxonomy.yaml)
	$(PY) -m aa_agent.cli taxonomy-show --config $(CONFIG)

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
	$(PY) scripts/freeze_cache.py

clean:  ## Remove tool caches (pytest/ruff/mypy/__pycache__)
	@$(PY) scripts/clean.py
