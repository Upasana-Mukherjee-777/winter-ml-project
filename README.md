# TrustGate-PHM  
Adaptive Multimodal Remaining Useful Life Prediction with Reliability-Aware Fusion (CMAPSS FD002)

---

## Overview

This repository implements an interpretable, reliability-aware, multimodal Remaining Useful Life (RUL) prediction framework for Prognostics and Health Management (PHM), evaluated on the NASA C-MAPSS FD002 dataset.

The framework addresses key challenges in real-world PHM systems, including multimodal data fusion, varying operating conditions, uncertainty, and decision reliability. Instead of fixed or heuristic fusion strategies, the proposed TrustGate approach learns adaptive modality trust weights based on temporal reliability and operating context.

---

## Key Features

- Regression-based RUL prediction (no classification assumption)
- Multimodal learning using sensor, visual (PCA-based), and tabular features
- Temporal feature engineering (rolling statistics, slopes, degradation trends)
- Operating-condition–specific modeling (FD002)
- TrustGate neural network for adaptive fusion
- Temporal reliability estimation
- Modality contradiction detection
- Risk-aware maintenance decision support
- NASA asymmetric scoring metric
- SHAP-based model explainability
- Visual analytics and dashboard generation

---

## Dataset

- NASA C-MAPSS FD002
- Multiple operating regimes
- Significant domain shift across conditions

Expected directory structure:

data/

├── train_FD002.txt

├── test_FD002.txt

└── RUL_FD002.txt


---

## Project Structure

├── data/

│ └── FD002 files

│

├── src/

│ ├── data_loader.py

│ ├── preprocessing.py

│ ├── feature_engineering.py

│ ├── ablation_studies.py

│ ├── model_comparison.py

│ ├── trustgate.py

│ ├── condition_specific.py

│ ├── contradiction.py

│ ├── evaluation.py

│ └── dashboard.py

│
├── notebooks/

│ └── experiments.ipynb

│

├── requirements.txt

└── README.md

---

## Installation

Clone the repository and install dependencies


### Required Libraries

- numpy
- pandas
- scikit-learn
- torch
- matplotlib
- seaborn
- shap (optional, for explainability)
- xgboost (optional)

---

## Methodology Overview

### 1. Data Preparation
- Load FD002 data
- Remove invalid and low-variance sensors
- Compute true RUL per engine
- Apply three-sigma outlier capping

### 2. Feature Engineering
- Rolling mean and standard deviation
- Temporal slope features
- Cycle normalization
- PCA-based visual proxy features

### 3. Modality Definition
- Sensor modality: raw and temporal sensor features
- Visual modality: PCA latent degradation features
- Tabular modality: operating settings and normalized cycle

### 4. Condition-Specific Training
- Models trained separately per operating condition
- Random Forest regressors used for each modality

### 5. TrustGate Fusion
- Inputs: operating conditions, cycle position, average temporal reliability
- Output: softmax-normalized trust weights for each modality
- Objective: minimize MAE on validation RUL

### 6. Inference and Evaluation
- Adaptive fusion at the engine level
- Evaluation using MAE, NASA score, and risk-based metrics
- Explainability via SHAP analysis

---

## Contradiction Detection

Disagreement between modalities is quantified using the standard deviation of modality predictions. Predictions exceeding a defined disagreement threshold are flagged for review, enabling uncertainty-aware decision support.

---

## Risk-Aware Decision Support

Predicted RUL values are converted into failure risk scores using a logistic mapping function. These risk scores are used for maintenance prioritization and are visualized in the decision dashboard.

---

## Evaluation Metrics

- Mean Absolute Error (MAE)
- NASA asymmetric score
- Coefficient of determination (R²)
- Risk-based accuracy, precision, recall, and F1 score
- Modality contradiction rate

---

## Visualization and Explainability

- True vs predicted RUL plots
- Residual analysis
- Modality-wise performance comparison
- Adaptive trust weight distributions
- Risk score histograms
- SHAP summary plots for sensor modality

---

## Reproducibility Notes

- Fixed random seeds
- Per-engine normalization
- No data leakage across engines
- Validation-based TrustGate training

---

## Intended Use

- Academic research in PHM and RUL prediction
- Benchmarking adaptive fusion strategies
- Explainable AI studies for safety-critical systems
- Maintenance decision-support prototyping

---

## Citation

If you use this code, please cite:

Adaptive Multimodal Remaining Useful Life Prediction with Reliability-Aware Fusion on NASA C-MAPSS FD002

---

## Acknowledgements

- NASA PHM Challenge
- C-MAPSS Dataset
- SHAP library
