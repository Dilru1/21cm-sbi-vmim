# Simple task runner -- `make <target>`. A gentle on-ramp to automation:
# one memorable command per common action.
#
# Note on lint: `ruff check` exits non-zero whenever ANY finding remains, and
# the legacy research code still has minor advisory nits. So the fmt/lint
# targets below are deliberately NON-FATAL (the leading `-` tells make to
# ignore the exit code) -- formatting and reporting should never "fail the
# build". The real gate is `make test`. Tighten later by removing the `-`.
.PHONY: help setup lint fmt fix test hooks clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

setup:           ## install runtime + dev dependencies
	pip install -r requirements.txt
	pip install ruff pytest pre-commit

hooks:           ## install pre-commit git hooks (run once)
	pre-commit install

fmt:             ## auto-format the codebase (safe; whitespace only)
	ruff format .

fix:             ## apply ruff's safe autofixes (non-fatal; some nits remain by design)
	-ruff check --fix .

lint:            ## report remaining lint findings (advisory; not clean on legacy code yet)
	-ruff check .

test:            ## run the torch-free test suite (the real gate)
	pytest -q -m "not needs_torch and not needs_data"

clean:           ## remove python caches
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
