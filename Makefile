# ══════════════════════════════════════════════════════════════
#  DeepFake Detector — Makefile
# ══════════════════════════════════════════════════════════════

.PHONY: dev backend frontend docker clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Local Development ─────────────────────────────────────────

backend: ## Start backend (FastAPI + ML models)
	cd backend && python server_lite.py

frontend: ## Start frontend (Next.js dev server)
	cd frontend && npm run dev

dev: ## Start both backend and frontend
	@echo "Starting backend in background..."
	cd backend && python server_lite.py &
	@echo "Starting frontend..."
	cd frontend && npm run dev

# ── Docker ────────────────────────────────────────────────────

docker: ## Build and run with Docker Compose
	docker compose up --build

docker-backend: ## Build and run backend only
	docker compose up --build backend

# ── Setup ─────────────────────────────────────────────────────

setup-backend: ## Install backend Python dependencies
	cd backend && pip install -r requirements.txt

setup-frontend: ## Install frontend Node dependencies
	cd frontend && npm install

setup: setup-backend setup-frontend ## Install all dependencies

# ── Cleanup ───────────────────────────────────────────────────

clean: ## Remove build artifacts
	rm -rf frontend/.next frontend/out
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
