# Insurance ML System (WIP)

This repository demonstrates a **production-style machine learning workflow**
for an insurance cost prediction (regression) use case.

The focus of this project is **ML system design**, not model performance.

---

## Problem Statement

Internal data teams often struggle to move machine learning models from
notebooks into stable, reproducible, and deployable systems.

Common issues include:
- non-reproducible training runs
- unclear data versions
- fragile deployment setups
- lack of traceability between data, model, and metrics

This project shows how to structure an ML system to address these issues.

---

## Project Scope

This repository demonstrates:

- clear separation of raw and processed data
- script-based preprocessing (no notebook dependency)
- config-driven model training
- saved model artifacts and metrics
- a standalone inference layer suitable for APIs or batch jobs

The goal is to provide a **clean, repeatable baseline** for production ML systems.

---

## Repository Structure

insurance-ml-system/
├── data/
│ ├── raw/ # Raw, immutable input data
│ └── processed/ # Preprocessed, model-ready data
├── src/
│ ├── preprocess.py # Data preprocessing pipeline
│ ├── train.py # Config-driven training script
│ ├── inference.py # Local inference module
│ └── utils.py
├── configs/
│ └── train_config.yaml # Training configuration
├── notebooks/
│ └── exploration.ipynb # Lightweight data exploration
├── models/
│ ├── model.pkl # Trained model artifact
│ └── metrics.json # Evaluation metrics
├── requirements.txt
└── README.md


---

## Workflow Overview

1. **Raw data ingestion**
   - Public insurance dataset stored in `data/raw/`

2. **Preprocessing**
   - Script-based preprocessing
   - Categorical encoding
   - Output written to `data/processed/`

3. **Training**
   - Configuration-driven training
   - Model and metrics saved as artifacts

4. **Inference**
   - Standalone inference module
   - Designed to be wrapped by an API or batch pipeline

---

## Current Status

- [x] Project structure
- [x] Preprocessing pipeline
- [x] Config-driven training
- [x] Model artifact and metrics
- [x] Local inference module
- [ ] API deployment
- [ ] Monitoring and logging
- [ ] Data versioning integration
- [ ] Cloud deployment (Azure)

---

## Intended Use

This repository is intended to:
- demonstrate ML system design best practices
- serve as a foundation for deployment-ready ML workflows
- evolve into a reusable internal toolkit or paid offering

It is **not** intended as a plug-and-play production system.

---

## Disclaimer

This project is under active development and reflects an evolving approach
to production ML workflows.

