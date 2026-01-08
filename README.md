```markdown
# Concrete Strength Prediction & Uncertainty Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📌 Overview

This repository hosts a comprehensive machine learning project aimed at predicting the **Compressive Strength of Concrete**. Beyond standard prediction, this project emphasizes **rigorous hyperparameter optimization**, **model explainability**, and **uncertainty quantification**.

The goal is to provide not just a point prediction for concrete strength, but also to understand *why* the model makes a prediction and *how confident* we can be in that prediction using state-of-the-art statistical methods.

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Workflow & Methodology](#-workflow--methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Hyperparameter Tuning](#2-hyperparameter-tuning)
  - [3. Model Training](#3-model-training)
  - [4. Model Explainability](#4-model-explainability)
  - [5. Uncertainty Analysis](#5-uncertainty-analysis)
- [Results](#-results)
- [Usage](#-usage)

---

## 📂 Repository Structure

```text
concrete_data_analysis/
├── Data/                               # Contains training and testing datasets
│   ├── train.csv
│   └── test.csv
├── Hyperparameter_Tuning/              # Notebooks for various optimization strategies
│   ├── Hyperparameter_tuning.ipynb     # Comparison of Grid, Random, Bayes, etc.
│   └── output/                         # Logs and results of tuning processes
├── Hyperparameter tuning using Optuna/ # Deep dive into Optuna optimization
│   ├── Optuna_autosampler/             # Auto-sampling implementations
│   ├── Optuna_PGBM/                    # Tuning for Probabilistic Gradient Boosting
│   └── Optuna_2/                       # Advanced Optuna configurations
├── Model_Explainations/                # SHAP and LIME interpretations
│   └── Model_explainations.ipynb
├── Uncertainity_Analysis/              # Core uncertainty quantification modules
│   ├── Conformal_Predictions/          # MAPIE and PUNCC implementations
│   ├── Probabilistic _Distribution (IBUG)/ # NGBoost & PGBM probabilistic outputs
│   └── Quantile_Regression/            # Gradient Boosting Quantile Regression
└── README.md                           # Project documentation

```

---

## 📊 Dataset

The project utilizes a dataset containing physical and chemical properties of concrete to predict its compressive strength.

**Features:**

* **`C`**: Cement (kg/m³)
* **`mp`**: Mineral Admixtures / Slag (kg/m³)
* **`FA`**: Fine Aggregate (kg/m³) - *Note: Based on volume, could also refer to Fly Ash in some contexts.*
* **`CA`**: Coarse Aggregate (kg/m³)
* **`F`**: Fly Ash or Silica Fume (kg/m³)
* **`W_P`**: Water-to-Powder ratio or Water Content
* **`Adm`**: Superplasticizer (Admixture)
* **`str`**: **Target Variable** - Compressive Strength (MPa)

Data is split into `train.csv` and `test.csv` located in the `Data/` directory.

---

## ⚙️ Installation

To reproduce the results, ensure you have Python installed along with the required libraries. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/yourusername/concrete_data_analysis.git](https://github.com/yourusername/concrete_data_analysis.git)
cd concrete_data_analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install xgboost lightgbm catboost gpboost ngboost
pip install optuna hyperopt bayesian-optimization
pip install shap lime interpret mapie puncc

```

---

## 🚀 Workflow & Methodology

This project follows a structured data science pipeline designed for high reliability and interpretability.

### 1. Data Preprocessing

* **Normalization**: Z-score normalization (`StandardScaler`) is applied to features to ensure all inputs contribute equally to model training, particularly important for models like SVM and Linear Regression.
* **Splitting**: Data is pre-split into training and testing sets to prevent data leakage.

### 2. Hyperparameter Tuning

We employ multiple advanced strategies to find the optimal configuration for each model. The `Hyperparameter_Tuning` folder explores:

* **Grid Search**: Exhaustive search over specified parameter values.
* **Random Search**: Random sampling of parameter settings.
* **Bayesian Optimization**: Probabilistic model-based optimization to find the minimum of the objective function.
* **Optuna**: The primary framework used for efficient, automated hyperparameter optimization. Specific focus is placed on `Optuna` in dedicated folders (`Optuna_1`, `Optuna_2`, `Optuna_autosampler`) to fine-tune boosting algorithms.

### 3. Model Training

A wide variety of regression algorithms are trained and compared:

* **Linear Models**: Linear Regression, SVM (SVR).
* **Tree-Based**: Decision Trees, Random Forest.
* **Boosting Algorithms**:
* **XGBoost** (Extreme Gradient Boosting)
* **LightGBM** (Light Gradient Boosting Machine)
* **CatBoost** (Categorical Boosting)
* **Gradient Boosting** (sklearn implementation)
* **GPBoost** (Gaussian Process Boosting)
* **NGBoost** (Natural Gradient Boosting)



### 4. Model Explainability

To overcome the "black-box" nature of complex models, we use `Model_Explainations/Model_explainations.ipynb`:

* **SHAP (SHapley Additive exPlanations)**: Provides global and local interpretability by calculating the contribution of each feature to the prediction. We use `TreeExplainer` for boosting models and `KernelExplainer` for others.
* **LIME (Local Interpretable Model-agnostic Explanations)**: Approximates the complex model locally with an interpretable one (linear model) to explain individual predictions.
* **InterpretML**: Used to visualize feature importance and generate dashboard-like explanations.

### 5. Uncertainty Analysis

This is a key differentiator of this project. Instead of just predicting *strength = 35 MPa*, we quantify the uncertainty of that prediction using three distinct approaches:

#### A. Conformal Prediction (`Uncertainity_Analysis/Conformal_Predictions`)

Uses the **MAPIE** and **PUNCC** libraries to generate prediction intervals with a guaranteed coverage rate (e.g., 90%).

* **Strategies**: Split Conformal Prediction, Cross-Validation+, Jackknife+-after-Bootstrap.
* **Output**: Intervals `[Lower, Upper]` that contain the true value with high probability.

#### B. Probabilistic Distribution (`Uncertainity_Analysis/Probabilistic_Distribution (IBUG)`)

Uses models like **NGBoost** and **PGBM** (Probabilistic Gradient Boosting Machines) that predict parameters of a probability distribution (mean  and standard deviation ) rather than a single point.

* Allows calculating the probability that the strength exceeds a certain threshold.

#### C. Quantile Regression (`Uncertainity_Analysis/Quantile_Regression`)

Models (like Gradient Boosting and LightGBM) are trained to minimize the Pinball Loss function for specific quantiles (e.g., 5th and 95th percentiles).

* This directly generates the lower and upper bounds of the prediction interval without assuming a specific distribution shape (like Gaussian).

---

## 📈 Results

Results, including predicted CSVs and metrics logs, are stored in the `output/` directories within each sub-module.

* **Best Performers**: Generally, Boosting models (CatBoost, XGBoost) combined with Optuna tuning yield the lowest RMSE.
* **Uncertainty**: Conformal prediction provides calibrated intervals that reliably capture the true concrete strength test results.

---

## 💡 Usage

To run the main analysis, launch Jupyter Notebook and open the relevant file:

**For Hyperparameter Tuning:**

```bash
jupyter notebook "Hyperparameter tuning using Optuna/Optuna_2/Hyperparameter_tuning_Optuna_2.ipynb"

```

**For Explainability:**

```bash
jupyter notebook "Model_Explainations/Model_explainations.ipynb"

```

**For Uncertainty Quantification:**

```bash
jupyter notebook "Uncertainity_Analysis/Conformal_Predictions/Conformal_Predictions(MAPIE,PUNCC).ipynb"

```

---

*Created by Danesh Selwal*

```

```
