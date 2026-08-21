# Simple task runner -- `make <target>`. A gentle on-ramp to automation:
# one memorable command per common action, no need to recall long invocations.
.PHONY: help setup lint fmt test hooks clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

setup:           ## install runtime + dev dependencies
	pip install -r requirements.txt
	pip install ruff pytest pre-commit

hooks:           ## install pre-commit git hooks (run once)
	pre-commit install

lint:            ## static checks
	ruff check .

fmt:             ## auto-format the codebase
	ruff format .
	ruff check --fix .

test:            ## run the torch-free test suite
	pytest -q -m "not needs_torch and not needs_data"

clean:           ## remove python caches
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
