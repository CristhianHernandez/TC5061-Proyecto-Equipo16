#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5061-proyecto-equipo16
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install project dependencies plus quality tools (PyTest, DeepChecks, Bandit, pip-audit)
.PHONY: requirements-dev
requirements-dev:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests/ -q

## Run only the training reproducibility tests (seed 16)
.PHONY: test-repro
test-repro:
	$(PYTHON_INTERPRETER) -m pytest tests/ -q -m reproducibilidad


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Start the MLflow tracking server (sqlite backend, artifacts in ./mlruns)
MLFLOW_PORT ?= 5000
.PHONY: mlflow
mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port $(MLFLOW_PORT)


# Todas las corridas se lanzan con el MLflow Project (MLproject) en el experimento del equipo.
# ENV_MANAGER=uv crea el entorno de python_env.yaml; ENV_MANAGER=local usa el entorno activo.
ENV_MANAGER ?= uv
EXPERIMENTO ?= mibici-equipo16
export MLFLOW_TRACKING_URI ?= http://127.0.0.1:$(MLFLOW_PORT)
MLFLOW_RUN = mlflow run . --env-manager $(ENV_MANAGER) --experiment-name $(EXPERIMENTO)
QUESTION ?= p1
MODEL ?=

## Ingesta de datos con el MLflow Project (entrada ingest)
.PHONY: data
data:
	$(MLFLOW_RUN) -e ingest

## Entrenar un modelo: make train QUESTION=p1 MODEL=hgb (sin MODEL entrena el baseline)
.PHONY: train
train:
	$(MLFLOW_RUN) -e train -P question=$(QUESTION) -P model=$(MODEL)

## Entrenar los tres baselines de referencia (P1, P2 y P3)
.PHONY: baselines
baselines:
	$(MLFLOW_RUN) -e train -P question=p1
	$(MLFLOW_RUN) -e train -P question=p2
	$(MLFLOW_RUN) -e train -P question=p3

## Registrar y promover a producción el mejor modelo: make register QUESTION=p1
.PHONY: register
register:
	$(MLFLOW_RUN) -e register -P question=$(QUESTION)

## Predecir con el modelo en producción (alias champion): make predict QUESTION=p1
.PHONY: predict
predict:
	$(MLFLOW_RUN) -e predict -P question=$(QUESTION)

## Tablas comparativas de P1, P2 y P3 en reports/ (fuera de Git)
.PHONY: compare
compare:
	$(MLFLOW_RUN) -e compare


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)