.PHONY: install test lint generate train evaluate api ui docker

install:
	python3.11 -m venv .venv
	.venv/bin/pip install -e ".[dev,model]"

test:
	.venv/bin/pytest --cov=documind --cov-report=term-missing

lint:
	.venv/bin/ruff check .

generate:
	.venv/bin/python scripts/generate_invoices.py --count 30 --anomaly-every 10

train:
	.venv/bin/python scripts/generate_invoices.py --count 180 --scan-every 0 --seed 17 --output training/generated
	.venv/bin/python scripts/train_ranker.py

evaluate:
	.venv/bin/python evaluation/evaluate.py --provider all

api:
	.venv/bin/uvicorn documind.main:app --reload

ui:
	.venv/bin/streamlit run frontend/app.py

docker:
	docker compose up --build
