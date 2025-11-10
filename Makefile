# Makefile for the Fraud Detection System

# --- Cross-Platform Virtual Environment Setup ---
# This checks the Operating System and sets the correct path
ifeq ($(OS),Windows_NT)
    # Windows paths
    PYTHON = venv\Scripts\python.exe
    ACTIVATE = venv\Scripts\activate.bat
else
    # Linux/macOS paths
    PYTHON = venv/bin/python
    ACTIVATE = source venv/bin/activate
endif
# --- End of Setup ---

.PHONY: all install process-data train mlflow-ui docker-build docker-up docker-down

# Default command
all: process-data train

# Target to create venv and install dependencies
# This now depends on the OS-specific python path
install: $(PYTHON)
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# This rule creates the venv if $(PYTHON) doesn't exist
$(PYTHON):
	@echo "Creating virtual environment..."
	python -m venv venv

# Target to run the data processing pipeline
process-data: install
	@echo "Running data processing pipeline..."
	$(PYTHON) src/run_data_processing.py
	
# Target to run the model training pipeline
train: install
	@echo "Running model training pipeline..."
	$(PYTHON) src/run_model_training.py
	
# Target to launch the MLflow UI
mlflow-ui: install
	@echo "Launching MLflow UI... (Access at http://127.0.0.1:5000)"
	@echo "This command will run in the foreground. Press Ctrl+C to stop."
	$(PYTHON) -m mlflow ui

# --- Docker Commands (for Airflow & MLflow) ---

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker services (MLflow & Airflow)..."
	@echo "Airflow UI: http://localhost:8080 (admin/admin)"
	@echo "MLflow UI:  http://localhost:5000"
	docker-compose up -d --build

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down