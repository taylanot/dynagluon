# DynaGluon

> [!WARNING]
> This project is currently in active development. Features and APIs are subject to change without notice.

DynaGluon is an AutoML engine and command-line interface (CLI) built on top of [AutoGluon Tabular](https://auto.gluon.ai/) and [Dynaconf](https://www.dynaconf.com/). It simplifies the process of training, extending, refitting, and predicting with AutoGluon models through simple YAML configuration files.

## Features

- **Train**: Train robust AutoML models using YAML configurations.
- **Extend**: Expand an existing model with additional algorithms or hyperparameter search trials.
- **Refit**: Refit trained models on 100% of the training data for final production deployment.
- **Predict**: Run inference on new data and save predictions seamlessly.
- **Configuration-Driven**: Easily manage experiments, model parameters, and environments without changing code.

## Requirements

- Python 3.9, 3.10, or 3.11
- [Pixi](https://pixi.sh/) (recommended for dependency and task management)

## Installation

The recommended way to install and manage the project is using `pixi`:

```bash
# Install dependencies and set up the environment
pixi install
```

Alternatively, you can install the package in editable mode using `pip`:

```bash
pip install -e .
```

## Usage

DynaGluon provides the `dynagluon` CLI tool. You can view all available commands by running `dynagluon --help`.

### Training a Model
Train a new AutoGluon model using your training data and a configuration file:
```bash
dynagluon train --config config.yaml --data train_data.csv
```

### Predicting
Run inference using a previously trained model on new test data:
```bash
dynagluon predict --config config.yaml --model-path AutogluonModels/your_model_dir --data test_data.csv --output predictions.csv
```

### Extending a Model
Continue training or add more search trials/models to an existing trained directory:
```bash
dynagluon extend --model-path AutogluonModels/your_model_dir --config config_extend.yaml
```

### Refitting a Model
Refit the best model (or a specified model) on all available data before deploying it to production:
```bash
dynagluon refit --model-path AutogluonModels/your_model_dir
```

## Development

The project uses `pixi` for task execution, linting, and testing.

**Run tests:**
```bash
pixi run test
```

**Linting and Formatting:**
```bash
pixi run lint       # Run ruff check
pixi run format     # Run ruff format
pixi run typecheck  # Run type checking
```
