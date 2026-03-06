.PHONY: setup ingest train serve demo

# Default month if not provided
MONTH ?= 2025-01

setup:
	pip install -r requirements.txt
	cd feature_repo && feast apply

ingest:
	python pipelines/ingest_data.py --month $(MONTH)

train:
	python pipelines/train_model.py

serve:
	uvicorn api.main:app --reload --port 8000

demo:
	bash scripts/demo_failover.sh
