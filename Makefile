.PHONY: build up down restart logs shell

export UID := $(shell id -u)
export GID := $(shell id -g)

IMAGE_NAME=ml-streamlit-portfolio-ml_portfolio
CONTAINER_NAME=python_ds_app
SERVICE_NAME=ml_portfolio

help:
	@printf "\033[33mUsage:\033[0m\n  make [target] [arg=\"val\"...]\n\n\033[33mTargets:\033[0m\n"
	@grep -E '^[-a-zA-Z0-9_\.\/]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build env (first command after project pulling)
	docker-compose build

up: ## Run containers
	docker-compose up -d
	@echo "🚀 Container is successfully spun up!"
	@echo "📝 Check out JupyterLab at: http://localhost:8888"
	@echo "🌧️ Streamlit is ready on: http://localhost:8501"

down: ## Stop containers (take a break, last command after deal)
	docker-compose down

restart: ## Restart
	docker-compose down && docker-compose up -d

logs: ## Check last logs
	docker-compose logs -f

shell: ## Log in to main container
	docker exec -it ${CONTAINER_NAME} /bin/bash
