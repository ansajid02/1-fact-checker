# 1-fact-checker: one command per job.   make help
PY        = uv run python
CLAIM    ?= The Burj Khalifa was completed in 2015.
MODE     ?= single
DATASET  ?= fever
LIMIT    ?=
WORKERS  ?= 4
TIER     ?= flash
BACKEND  ?=
JUDGE    ?= llm
SEARCHERS?= wiki,kb,papers
FILTER   ?= gemini
JUDGES   ?= llm:flash ollama:$(BASE_MODEL) ollama:factjudge
BASE_MODEL ?= hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
LLM_FLAGS = --tier $(TIER) --judge $(JUDGE) --searchers $(SEARCHERS) $(if $(BACKEND),--backend $(BACKEND))

.PHONY: help setup smoke models data kb kb-bm25 kb-eval check check-single bench compare lora-data \
        ollama-base ollama-judge judge-eval trace test

help:
	@echo "Setup"
	@echo "  make setup                      install deps, create .env and prices.toml"
	@echo "  make smoke [TIER=pro]           one structured LLM call (checks key + model)"
	@echo "  make models [FILTER=gemini]     list models for the current backend"
	@echo "Data"
	@echo "  make data                       FEVER test/train samples + AraFacts sample"
	@echo "  make kb                         knowledge base: 1.7 GB dump -> pages -> chunks -> embeddings"
	@echo "  make kb-bm25                    same, without embeddings (free, BM25 only)"
	@echo "  make kb-eval [RERANK=1]         retrieval recall on the test claims"
	@echo "Use"
	@echo "  make check CLAIM=\"...\"          agent team -> report -> your approval -> published/"
	@echo "  make check-single CLAIM=\"...\"   the single-agent baseline"
	@echo "Benchmark"
	@echo "  make bench MODE=single|team [DATASET=fever|arafacts] [LIMIT=30] [JUDGE=llm|llm:pro|lora:factjudge]"
	@echo "  make compare                    table of all runs -> results/RESULTS.md + README"
	@echo "LoRA judge (step 11)"
	@echo "  make lora-data                  training data + fixed-evidence judge test set"
	@echo "  (train in Colab: lora/train_judge.ipynb, put factjudge-Q4_K_M.gguf in lora/)"
	@echo "  make ollama-base                pull the base model into the shared Ollama install"
	@echo "  make ollama-judge               register the LoRA model in Ollama as 'factjudge'"
	@echo "  make judge-eval [JUDGES=...]    compare judges on the same evidence"
	@echo "Other"
	@echo "  make trace RUN=<run_id>         summarise a trace"
	@echo "  make test                       offline unit tests"

setup:
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env: fill in AWS_BEARER_TOKEN_BEDROCK and HTTP_USER_AGENT")
	@test -f prices.toml || (cp prices.example.toml prices.toml && echo "Created prices.toml (only needed for Vertex AI)")

smoke:
	$(PY) -m agentcore.smoke $(TIER)

models:
	$(PY) -m agentcore.models $(FILTER) $(BACKEND)

data:
	$(PY) -m factcheck prepare-data

kb:
	$(PY) -m factcheck kb

kb-bm25:
	$(PY) -m factcheck kb --no-embed

kb-eval:
	$(PY) -m factcheck kb-eval $(if $(RERANK),--rerank)

check:
	$(PY) -m factcheck check "$(CLAIM)" --mode team $(LLM_FLAGS)

check-single:
	$(PY) -m factcheck check "$(CLAIM)" --mode single $(LLM_FLAGS)

bench:
	$(PY) -m factcheck bench --mode $(MODE) --dataset $(DATASET) --workers $(WORKERS) $(LLM_FLAGS) $(if $(LIMIT),--limit $(LIMIT))

compare:
	$(PY) -m factcheck compare

lora-data:
	$(PY) -m factcheck lora-data

# Ollama is installed ONCE system-wide and shared by all projects (models live in ~/.ollama).
ollama-base:
	@command -v ollama >/dev/null || (echo "Install Ollama once (https://ollama.com/download), then rerun." && exit 1)
	ollama pull $(BASE_MODEL)

ollama-judge:
	@command -v ollama >/dev/null || (echo "Install Ollama once (https://ollama.com/download), then rerun." && exit 1)
	@test -f lora/factjudge-Q4_K_M.gguf || (echo "Put factjudge-Q4_K_M.gguf (from the Colab notebook) in lora/ first." && exit 1)
	cd lora && ollama create factjudge -f Modelfile

judge-eval:
	$(PY) -m factcheck judge-eval --judges "$(JUDGES)" $(if $(LIMIT),--limit $(LIMIT))

trace:
	$(PY) -m factcheck trace $(RUN)

test:
	uv run pytest -q
