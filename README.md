# 🌊 ML Analysis of Scour Data: Uncertainty Quantification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

A comprehensive machine learning repository dedicated to predicting **Scour Depth** around bridge piers. This project goes beyond simple point prediction by incorporating rigorous **Hyperparameter Tuning** and advanced **Uncertainty Quantification (UQ)** techniques to ensure robust and reliable hydraulic engineering applications.

---

## 📑 Table of Contents (Navigation)

1. [📌 Project Overview](#-project-overview)
2. [📂 Repository Structure](#-repository-structure)
3. [📊 Dataset Details](#-dataset-details)
4. [🛠️ Workflow & Methodology](#-workflow--methodology)
    - [Phase 1: Hyperparameter Tuning](#phase-1-hyperparameter-tuning)
    - [Phase 2: Quantile Regression](#phase-2-quantile-regression)
    - [Phase 3: Probabilistic Distribution](#phase-3-probabilistic-distribution)
    - [Phase 4: Conformal Predictions](#phase-4-conformal-predictions)

---

## 📌 Project Overview

This repository implements a full data science pipeline for analyzing scour data. It addresses the following key challenges in hydraulic structure safety:
-   **Optimization**: Utilizing **Optuna** to automatically find the optimal hyperparameters for complex gradient boosting models.
-   **Robustness**: Modeling Aleatoric uncertainty to understand the inherent variability in scour depth.
-   **Reliability**: Estimating prediction intervals using **Conformal Prediction** and **Quantile Regression** to provide guaranteed coverage (e.g., 90% confidence).

---

## 📂 Repository Structure

The repository is organized into five main modules. Click on the headings to navigate to the files.

### 1. 🗂️ [Data](./Data)
Contains the hydraulic dataset split into training and testing sets.
-   **[`train.csv`](./Data/train.csv)**: Historical experimental data used for model training.
-   **[`test.csv`](./Data/test.csv)**: Unseen data used for final evaluation and validation.

### 2. 🎛️ [Hyperparameter Tuning](./Hyperparameter%20Tuning)
We employ advanced Bayesian optimization to maximize model performance.
-   **[`Optuna_autosampler_scour.ipynb`](./Hyperparameter%20Tuning/Optuna_autosampler_scour.ipynb)**:
    -   Implements **Optuna** with the **Tree-structured Parzen Estimator (TPE)** sampler.
    -   Optimizes regressors like **XGBoost**, **LightGBM**, **CatBoost**, and **NGBoost** to minimize RMSE.
-   **[`test_results.xlsx`](./Hyperparameter%20Tuning/test_results.xlsx)**: Comprehensive logs of the best parameters found during the tuning sessions.

### 3. 📉 [Quantile Regression](./Quantile%20Regression)
A non-parametric approach to estimating prediction intervals.
-   **[`Quantile_Regression_scour.ipynb`](./Quantile%20Regression/Quantile_Regression_scour.ipynb)**:
    -   Trains models to predict conditional quantiles (e.g., $q_{0.05}$ and $q_{0.95}$) directly.
    -   Evaluates the "Pinball Loss" to measure the quality of the predicted intervals.

### 4. 🧠 [Probabilistic Distribution](./Probabilistic%20Distribution)
Treats the target variable as a distribution rather than a point estimate.
-   **[`Probabilistic__Distribution_scour.ipynb`](./Probabilistic%20Distribution/Probabilistic__Distribution_scour.ipynb)**:
    -   Uses **NGBoost** (Natural Gradient Boosting) and **PGBM** (Probabilistic Gradient Boosting Machines).
    -   Outputs the full probability density function (PDF), providing parameters $\mu$ (mean) and $\sigma$ (standard deviation) for each prediction.

### 5. 🛡️ [Conformal Predictions](./Conformal%20Predictions)
The most rigorous statistical layer for engineering safety.
-   **[`Conformal Predictions(MAPIE,PUNCC)_scour.ipynb`](./Conformal%20Predictions/Conformal%20Predictions(MAPIE,PUNCC)_scour.ipynb)**:
    -   Leverages **MAPIE** and **PUNCC** libraries to generate prediction intervals with mathematically guaranteed coverage.
    -   Implements methods like **Split Conformal Prediction (SCP)** and **CQR (Conformalized Quantile Regression)**.

---

## 📊 Dataset Details

The dataset comprises hydraulic and geometric parameters of bridge piers and sediment beds. The target variable is **Scour Depth**.

| Feature | Description | Unit |
| :--- | :--- | :--- |
| **Ps** | Pier Shape Factor | Dimensionless |
| **Pw** | Pier Width | m |
| **Skew** | Angle of Attack | Degrees (°) |
| **Velocity** | Flow Velocity | m/s |
| **Depth** | Flow Depth | m |
| **D50** | Median Sediment Size | mm |
| **Gradation** | Sediment Gradation Coefficient | Dimensionless |
| **Scour** | **Scour Depth (Target)** | m |

---

## 🛠️ Workflow & Methodology

### Phase 1: Hyperparameter Tuning
Before building UQ models, we ensure our base regressors are optimal.
1.  **Selection**: We integrate Gradient Boosting frameworks (XGBoost, CatBoost, LightGBM, GPBoost).
2.  **Optimization**: We run **Optuna** trials to search the hyperparameter space (learning rates, max depth, estimators).
3.  **Outcome**: The configuration with the lowest RMSE is saved and used as the backbone for the subsequent uncertainty modules.

### Phase 2: Quantile Regression
We move beyond the mean.
-   Instead of predicting just the expected scour depth, we predict the **5th and 95th percentiles**.
-   This provides a "Likely Range" of scour, allowing engineers to account for worst-case scenarios without assuming a specific data distribution.

### Phase 3: Probabilistic Distribution
We model the aleatoric uncertainty.
-   **Method**: We assume the scour depth follows a parametric distribution (e.g., Normal or Laplace).
-   **Output**: The model returns a mean ($\mu$) and variance ($\sigma^2$) for every input. A high $\sigma$ alerts engineers to high uncertainty in that specific hydraulic condition.

### Phase 4: Conformal Predictions
We provide statistical guarantees.
-   **Calibration**: We use a calibration set (subset of `train.csv`) to adjust our intervals.
-   **Guarantee**: The generated intervals `[Lower, Upper]` are mathematically guaranteed to contain the true scour depth with a probability of $1-\alpha$ (e.g., 90%), assuming the data is exchangeable.

---
