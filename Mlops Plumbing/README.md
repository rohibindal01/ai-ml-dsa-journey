# ml-project-template

[![CI](https://github.com/your-org/ml-project-template/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/ml-project-template/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/your-org/ml-project-template/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/ml-project-template)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A production-ready, end-to-end machine learning project scaffold. Clone it, point it at your data, and ship.

---

## Architecture

```
+-------------+     +--------------+     +---------------+
|  Raw Data   |---->| Preprocessing|---->|    Features   |
|  (DVC)      |     |  split/clean |     |  engineering  |
+-------------+     +--------------+     +-------+-------+
                                                 |
                                        +--------v--------+
                                        | Model Training  |
                                        | MLflow + DVC    |
                                        +--------+--------+
                                                 |
                        +------------------------v--------+
                        |       MLflow Registry           |
                        +------------------------+--------+
                                                 |
                                        +--------v--------+
                                        |  FastAPI        |
                                        |  /predict       |
                                        +-----------------+
```

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/your-org/ml-project-template.git
cd ml-project-template

# 2. Install dependencies
make install

# 3. Install pre-commit hooks
make hooks

# 4. Copy and edit environment variables
cp .env.example .env

# 5. Start MLflow tracking server + API
make docker-up

# 6. Run the full DVC pipeline (preprocess -> train -> evaluate)
make pipeline

# 7. Run the test suite
make test

# 8. Open MLflow UI
open http://localhost:5000
```

## Project Structure

```
ml-project-template/
├── .github/workflows/   CI (lint + test) and CD (Docker push)
├── configs/             Hydra/OmegaConf YAML configuration
├── data/                Raw (DVC-tracked) and processed splits
├── notebooks/           EDA and exploration notebooks
├── src/
│   ├── data/            Ingestion and preprocessing
│   ├── features/        sklearn-compatible transformers
│   ├── models/          Abstract BaseModel + sklearn trainer
│   ├── evaluation/      Metrics, plots, reports
│   └── serving/         FastAPI prediction API
├── tests/               pytest test suite (>=80% coverage)
├── Dockerfile           Multi-stage build, non-root user
├── docker-compose.yml   MLflow + API services
├── dvc.yaml             Reproducible ML pipeline stages
├── Makefile             Developer shortcuts
└── pyproject.toml       Poetry deps, ruff, mypy config
```

## Configuration

All hyperparameters live in `configs/default.yaml` and are managed by [Hydra](https://hydra.cc).
Override any value on the command line:

```bash
python -m src.models.trainer model.type=gradient_boosting model.gradient_boosting.n_estimators=300
```

## MLOps Stack

| Concern | Tool |
|---|---|
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Config management | Hydra + OmegaConf |
| Serving | FastAPI + Uvicorn |
| Containerisation | Docker (multi-stage) |
| Linting | Ruff |
| Type checking | Mypy (strict) |
| Testing | pytest + pytest-cov |
| CI/CD | GitHub Actions |

## API Reference

### `GET /health`

Returns service health and model load status.

### `POST /predict`

```json
{
  "features": [
    {"mean_radius": 14.0, "mean_texture": 19.0}
  ]
}
```

Returns predictions and class probabilities.

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes with tests and type annotations.
3. Ensure all checks pass: `make check && make test`
4. Submit a pull request against `main`.

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/).
Pre-commit hooks enforce formatting and type safety automatically.

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
