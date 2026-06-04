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

DATASETS_DIR      := datasets_public/public
UNANSW_DIR        := $(DATASETS_DIR)/UnansweredQuestions
ANSW_DIR          := $(DATASETS_DIR)/AnsweredQuestions
OUT_SEARCH        := data/output/search_results
OUT_ANSWER        := data/output/search_results_and_answer

# LLM for generation (alt: HuggingFaceTB/SmolLM2-135M-Instruct)
MODEL       ?= Qwen/Qwen3-0.6B
# Embedding model for semantic search
EMBED_MODEL ?= BAAI/bge-small-en-v1.5
# Retrieval method: bm25 | embedding | hybrid
RETRIEVER   ?= hybrid
# Repository to index
REPO        ?= data/raw/vllm-0.10.1
# Top-k results per query
K           ?= 5
# Max characters per chunk / context window
CHUNK_SIZE  ?= 2000
# Max questions to process (0 = all)
LIMIT       ?= 0
# Query expansion: empty = auto, True = force on, False = force off
EXPAND      ?=

_EXPAND_FLAG := $(if $(EXPAND),--expand $(EXPAND),)

RUN   := $(UV) run python src/main.py

run: index search_docs search_code answer_all evaluate_docs evaluate_code

run-docs: index search_docs answer_docs evaluate_docs

run-code: index search_code answer_code evaluate_code

# Run the entry point under Python's debugger (e.g. make debug ARGS="search foo")
ARGS ?=
debug:
	@$(UV) run python -m pdb src/main.py $(ARGS)

index:
	@$(RUN) index \
		--repo_path $(REPO) \
		--retriever $(RETRIEVER) \
		--embedding_model $(EMBED_MODEL) \
		--max_chunk_size $(CHUNK_SIZE)

search_docs:
	@$(RUN) search_dataset \
		--dataset_path $(UNANSW_DIR)/dataset_docs_public.json \
		--save_directory $(OUT_SEARCH) \
		--retriever $(RETRIEVER) \
		--embedding_model $(EMBED_MODEL) \
		--k $(K) \
		--max_chunk_size $(CHUNK_SIZE) \
		--limit $(LIMIT) $(_EXPAND_FLAG)

search_code:
	@$(RUN) search_dataset \
		--dataset_path $(UNANSW_DIR)/dataset_code_public.json \
		--save_directory $(OUT_SEARCH) \
		--retriever $(RETRIEVER) \
		--embedding_model $(EMBED_MODEL) \
		--k $(K) \
		--max_chunk_size $(CHUNK_SIZE) \
		--limit $(LIMIT) $(_EXPAND_FLAG)

answer_docs:
	@$(RUN) answer_dataset \
		--student_search_results_path $(OUT_SEARCH)/dataset_docs_public.json \
		--save_directory $(OUT_ANSWER) \
		--model_name $(MODEL) \
		--max_chunk_size $(CHUNK_SIZE)

answer_code:
	@$(RUN) answer_dataset \
		--student_search_results_path $(OUT_SEARCH)/dataset_code_public.json \
		--save_directory $(OUT_ANSWER) \
		--model_name $(MODEL) \
		--max_chunk_size $(CHUNK_SIZE)

answer_all:
	@$(RUN) answer_all \
		--search_results_dir $(OUT_SEARCH) \
		--save_directory $(OUT_ANSWER) \
		--model_name $(MODEL) \
		--max_chunk_size $(CHUNK_SIZE)

evaluate_docs:
	@$(RUN) evaluate \
		--student_answer_path $(OUT_SEARCH)/dataset_docs_public.json \
		--dataset_path $(ANSW_DIR)/dataset_docs_public.json

evaluate_code:
	@$(RUN) evaluate \
		--student_answer_path $(OUT_SEARCH)/dataset_code_public.json \
		--dataset_path $(ANSW_DIR)/dataset_code_public.json

_EXCLUDE := .venv,__pycache__,moulinette_pkg,vllm-0.10.1,data

lint:
	@$(UV) run flake8 src/ --extend-exclude=$(_EXCLUDE)
	@$(UV) run mypy src/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--exclude 'data/'

lint-strict:
	@$(UV) run flake8 src/ --extend-exclude=$(_EXCLUDE)
	@$(UV) run mypy src/ --strict --exclude 'data/'

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache

# Like clean, but also wipe generated data (built indexes & outputs).
# Keeps data/raw (the source repo symlink).
fclean: clean
	rm -rf data/processed data/output

.PHONY: install run run-docs run-code debug index search_docs search_code answer_docs answer_code answer_all evaluate_docs evaluate_code lint lint-strict clean fclean
