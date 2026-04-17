#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops
PYTHON_VERSION = 3.11.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


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


## Build Docker image
.PHONY: build
build:
	docker build -t $(PROJECT_NAME):latest .


## Up Docker compose services
.PHONY: up
up: build
	docker compose up -d


## Down Docker compose services
.PHONY: down
down:
	docker compose down


## Fully remove Docker compose services
.PHONY: rm-docker
rm-docker: down
	-docker image rm $(PROJECT_NAME):latest


## Repro DVC pipeline
.PHONY: repro
repro:
	dvc repro


## Run tests
.PHONY: test
test:
	python -m pytest -q tests/


## Invokes /predict endpoint with fraud operation params
.PHONY: predict-fraud
predict-fraud:
	curl -sS -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{ \
		"Time": 35942.0, \
		"V1": -4.19407367570647, \
		"V2": 4.3828973624446705, \
		"V3": -5.11836337307965, \
		"V4": 4.45522984017686, \
		"V5": -4.81262068168969, \
		"V6": -1.22464470658392, \
		"V7": -7.28132786701771, \
		"V8": 3.33224979453268, \
		"V9": -3.6796585127756, \
		"V10": -7.52436833681058, \
		"V11": 2.95434436554049, \
		"V12": -7.09982534281257, \
		"V13": 1.52036890797685, \
		"V14": -7.68780276684597, \
		"V15": -0.225002021021292, \
		"V16": -8.52085016639749, \
		"V17": -13.2772997431934, \
		"V18": -5.25370461976771, \
		"V19": 3.6233318140275497, \
		"V20": 0.579098497681141, \
		"V21": 1.55047296606276, \
		"V22": 0.614572855219804, \
		"V23": 0.0285207917962781, \
		"V24": 0.0137043244720008, \
		"V25": -0.1495120128569549, \
		"V26": -0.131686656095248, \
		"V27": 0.4739342218325089, \
		"V28": 0.47375748155095, \
		"Amount": 14.46 \
		}'


## Invokes /predict endpoint with no-fraud operation params
.PHONY: predict-nofraud
predict-nofraud:
	curl -sS -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{ \
		"Time": 69510.0, \
		"V1": 1.04987705786529, \
		"V2": -0.0440822074351266, \
		"V3": 1.03769245792612, \
		"V4": 0.9215263360372191, \
		"V5": -0.429641491722005, \
		"V6": 0.711063656522172, \
		"V7": -0.6674831272030439, \
		"V8": 0.43287092356494605, \
		"V9": 0.21863399262208896, \
		"V10": 0.0400975124319757, \
		"V11": 1.6870704423107201, \
		"V12": 0.910455514423777, \
		"V13": -0.314160092682072, \
		"V14": 0.36583007572197707, \
		"V15": 1.2082045602262599, \
		"V16": 0.18033871643416602, \
		"V17": -0.32022343991278496, \
		"V18": -0.18448685793606798, \
		"V19": -1.00070959700381, \
		"V20": -0.18566521671975603, \
		"V21": 0.12474058148602499, \
		"V22": 0.394980166558275, \
		"V23": 0.0748753453711311, \
		"V24": -0.30578820486468294, \
		"V25": 0.141395370719422, \
		"V26": -0.4065128453280179, \
		"V27": 0.0821599962808053, \
		"V28": 0.0219767084587138, \
		"Amount": 10.68 \
		}'


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mlops/dataset.py


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
