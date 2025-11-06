# Makefile for the Fraud Detection System

# Define the Python executable from the venv
# This is the robust way to use a venv with Make
# Using backslashes for Windows
PYTHON = .venv\Scripts\python.exe

# Phony targets aren't real files
.PHONY: all install process-data train mlflow-ui

# 'all' will first process data, then train
all: process-data train

# This target creates the venv (if needed) and installs packages
# It depends on the venv marker file "$(PYTHON)"
install: $(PYTHON)
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -r requirements.txt

# This "sentinel" target checks if the venv exists.
# If $(PYTHON) (venv\Scripts\python.exe) doesn't exist, Make runs this
# command to create it. If it *does* exist, this is skipped.
$(PYTHON):
	@echo "Creating virtual environment..."
	python -m venv .venv

# Target to run the data processing pipeline
process-data:
	@echo "Running data processing pipeline..."
	$(PYTHON) src/run_data_processing.py
	
# Target to run the model training pipeline
train:
	@echo "Running model training pipeline..."
	$(PYTHON) src/run_model_training.py
	
# Target to launch the MLflow UI
mlflow-ui:
	@echo "Launching MLflow UI... (Access at http://127.0.0.1:5000)"
	$(PYTHON) -m mlflow ui