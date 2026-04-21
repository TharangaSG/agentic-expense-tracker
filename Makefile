.PHONY: help chainlit whatsapp install clean test

export PYTHONPATH := $(shell pwd)

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

chainlit: ## Run the Chainlit app
	uv run chainlit run src/interfaces/chainlit/app.py --watch

whatsapp: ## Run the WhatsApp bot
	uv run python run_whatsapp.py

install: ## Install dependencies
	uv sync

test: ## Run tests
	uv run pytest tests/

clean: ## Clean Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
