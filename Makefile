.PHONY: help install setup sim mock season season-mock smoke view replay-check clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install Python + Node dependencies
	pip install -r requirements.txt
	cd open-los && npm install

setup: ## Full setup: clone submodules, install deps
	git submodule update --init --recursive
	$(MAKE) install

update: ## Pull latest code and update submodules
	git pull
	git submodule update --init --recursive
	cd open-los && npm install

sim: ## Run full LOS simulation (realistic mix)
	./run_sim.sh

mock: ## Run mock simulation (no API key, no LOS)
	python -m loanville --mock --allow-non-los-formal

season: ## Run multi-week season via LOS
	./run_sim.sh --season

season-mock: ## Run multi-week season in mock mode
	python -m loanville --season --mock --allow-non-los-formal

smoke: ## Run smoke test (1 week, 3 borrowers per model)
	python run_smoke_test.py

view: ## Start web viewer (rebuilds season index)
	python -m loanville view

replay-check: ## Verify mock determinism (hash comparison)
	python scripts/check_mock_replay.py

clean: ## Remove generated/cached files
	rm -rf __pycache__ loanville/__pycache__ runs/replay web/seasons/index.json
