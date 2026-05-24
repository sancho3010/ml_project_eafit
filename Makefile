PYTHON = ../.venv/bin/python
SRC    = src tests

## Formatear código
.PHONY: format
format:
	$(PYTHON) -m ruff check --fix $(SRC)
	$(PYTHON) -m ruff format $(SRC)

## Lint (falla si hay issues)
.PHONY: lint
lint:
	$(PYTHON) -m ruff format --check $(SRC)
	$(PYTHON) -m ruff check $(SRC)

## Análisis de seguridad
.PHONY: security
security:
	$(PYTHON) -m bandit -r src -ll

## Tests unitarios
.PHONY: test
test:
	$(PYTHON) -m pytest tests -q

## Lint + seguridad + tests
.PHONY: check
check: lint security test
