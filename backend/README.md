# Healthcare Data Generation System

A comprehensive system for generating synthetic healthcare data with multi-diagnosis support for ICU patient research.

## Project Structure

This project is organized into modular components for easy maintenance and debugging:

- `config/`: Configuration files and target distributions
- `src/`: Main source code organized by functionality
- `scripts/`: Execution scripts for the complete pipeline
- `tests/`: Unit tests for all components
- `notebooks/`: Jupyter notebooks for analysis
- `data/`: Data storage (raw, processed, final)
- `docs/`: Documentation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete pipeline:
```bash
python scripts/01_setup_data.py
python scripts/02_process_synthea.py
python scripts/03_process_mimic.py
python scripts/04_combine_datasets.py
python scripts/05_validate_dataset.py
python scripts/06_train_ctgan.py
python scripts/07_evaluate_model.py
python scripts/08_run_api.py
```

## Features

- Multi-diagnosis support for complex patient conditions
- Clinical literature-based target distributions
- Comprehensive validation framework
- CTGAN-based synthetic data generation
- RESTful API for data querying
- Modular, testable codebase

## Documentation

See the `docs/` directory for detailed documentation on:
- API usage
- Validation methodology
- Deployment guidelines

Created on: 1752775387.5332694
