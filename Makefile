PYTHON_VERSION := $(shell cat .python-version)

_UV_DIRECT := $(wildcard $(HOME)/.local/bin/uv $(HOME)/.cargo/bin/uv /usr/local/bin/uv)
UV := $(if $(_UV_DIRECT),$(firstword $(_UV_DIRECT)),\
      $(shell find $(HOME)/.pyenv/versions -maxdepth 3 -name uv -type f 2>/dev/null | head -1))

ifeq ($(UV),)
$(error uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif

-include .env
export HF_TOKEN

install:
	$(UV) python install $(PYTHON_VERSION)
	$(UV) sync --frozen

run:
	$(UV) run python src/main.py

lint:
	@$(UV) run flake8 . --extend-exclude=.venv,__pycache__
	@$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@$(UV) run flake8 . --extend-exclude=.venv,__pycache__
	@$(UV) run mypy . --strict

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache

.PHONY: install run lint lint-strict clean
