The following `README.md` provides a comprehensive guide to your repository, detailing its structure, the scientific workflows implemented, and instructions for newcomers.

---

# Concrete Strength Prediction & Uncertainty Analysis

## 📌 Overview

This repository contains a comprehensive machine learning project focused on predicting the **Compressive Strength of Concrete**. Beyond standard prediction, this project emphasizes **Hyperparameter Optimization**, **Model Explainability**, and rigorous **Uncertainty Quantification**.

The goal is to build robust models that not only predict strength with high accuracy but also provide confidence intervals and explanations for their predictions, which is critical for safety-sensitive domains like civil engineering.

## 📖 Table of Contents

* [Overview](https://www.google.com/search?q=%23-overview)
* [Repository Structure](https://www.google.com/search?q=%23-repository-structure)
* [Dataset Description](https://www.google.com/search?q=%23-dataset-description)
* [Workflow & Methodology](https://www.google.com/search?q=%23-workflow--methodology)
* [1. Hyperparameter Tuning](https://www.google.com/search?q=%231-hyperparameter-tuning)
* [2. Model Explainability](https://www.google.com/search?q=%232-model-explainability)
* [3. Uncertainty Analysis](https://www.google.com/search?q=%233-uncertainty-analysis)


* [Installation & Requirements](https://www.google.com/search?q=%23-installation--requirements)
* [Usage](https://www.google.com/search?q=%23-usage)

---

## 📂 Repository Structure

The project is organized into modular directories, each focusing on a specific aspect of the ML pipeline:

```text
├── Data/
│   ├── train.csv                # Training dataset
│   └── test.csv                 # Testing dataset
│
├── Hyperparameter_Tuning/
│   ├── Hyperparameter_tuning.ipynb  # General tuning (Grid, Random, Bayes, Hyperband)
│   └── output/                      # Saved prediction results from tuning
│
├── Hyperparameter tuning using Optuna/
│   ├── Optuna_1/                # Study on different Optuna Pruners/Samplers
│   ├── Optuna_2/                # Extended Optuna optimization strategies
│   ├── Optuna_autosampler/      # Automated hyperparameter search using OptunaHub
│   └── Optuna_PGBM/             # Specific tuning for Probabilistic Gradient Boosting
│
├── Model_Explainations/
│   └── Model_explainations.ipynb    # SHAP and LIME analysis for feature importance
│
└── Uncertainity_Analysis/
    ├── Conformal_Predictions/
    │   └── Conformal_Predictions(MAPIE,PUNCC).ipynb  # CP using MAPIE and PUNCC libraries
    ├── Probabilistic _Distribution (IBUG)/
    │   └── Probabilistic__Distribution.ipynb         # NGBoost & PGBM probabilistic forecasting
    └── Quantile_Regression/
        └── Quantile_Regression.ipynb                 # Interval prediction via Quantile Regression

```

---

## 📊 Dataset Description

The dataset contains various components of concrete and its age, which are used to predict its **Compressive Strength (`str`)**.

| Feature | Description | Unit |
| --- | --- | --- |
| **C** | Cement (component 1) | kg/m³ |
| **mp** | Blast Furnace Slag (component 2) | kg/m³ |
| **FA** | Fly Ash (component 3) | kg/m³ |
| **CA** | Coarse Aggregate (component 6) | kg/m³ |
| **F** | Fine Aggregate (component 7) | kg/m³ |
| **W_P** | Water-Powder Ratio | Ratio |
| **Adm** | Superplasticizer (component 5) | kg/m³ |
| **str** | **Compressive Strength (Target)** | MPa |

---

## ⚙️ Workflow & Methodology

### 1. Hyperparameter Tuning

We employ state-of-the-art optimization techniques to maximize model performance (RMSE, R2).

* **Libraries:** `Optuna`, `Hyperopt`, `Scikit-learn` (GridSearchCV, RandomizedSearchCV).
* **Algorithms Tuned:** XGBoost, LightGBM, CatBoost, Gradient Boosting, Random Forest, NGBoost, GPBoost.
* **Techniques:**
* **Bayesian Optimization:** Efficiently searches the hyperparameter space.
* **Pruners:** Uses algorithms like `Hyperband` and `MedianPruner` to stop unpromising trials early.
* **Samplers:** Comparisons of `TPESampler`, `CmaEsSampler`, `NSGAIIISampler`, and `RandomSampler`.



### 2. Model Explainability

To trust the "black box" models, we explain why specific predictions are made.

* **Global Explainability:**
* **SHAP (SHapley Additive exPlanations):** Visualizes the global impact of features like Cement and Water-Powder ratio on strength.
* **InterpretML:** Uses Generalized Additive Models (GAMs) to show feature contributions.


* **Local Explainability:**
* **LIME (Local Interpretable Model-agnostic Explanations):** Explains individual predictions to see which features increased or decreased the strength for a specific sample.



### 3. Uncertainty Analysis

This is a key differentiator of this project. Instead of just a single point prediction (e.g., "35 MPa"), we provide confidence intervals (e.g., "32-38 MPa with 90% confidence").

#### A. Conformal Predictions

* **Tools:** `MAPIE`, `PUNCC`.
* **Methods:**
* **Split Conformal Prediction (SCP):** Calibrates intervals on a hold-out set.
* **Cross-Validation+ (CV+):** More robust interval estimation using cross-validation.
* **CQR (Conformal Quantile Regression):** Adjusts intervals based on the difficulty of the input.


* **Metrics:** Coverage (validity), Mean Interval Width (sharpness).

#### B. Probabilistic Distributions

* **Models:** `NGBoost` (Natural Gradient Boosting), `PGBM` (Probabilistic Gradient Boosting Machines).
* **Approach:** The model predicts the parameters of a probability distribution (Mean  and Standard Deviation ) for every sample.
* **Metrics:**
* **NLL (Negative Log Likelihood):** Measures how well the predicted distribution fits the data.
* **CRPS (Continuous Ranked Probability Score):** A robust metric for probabilistic forecasts.



#### C. Quantile Regression

* **Models:** Gradient Boosting, CatBoost, LightGBM (using quantile loss objectives).
* **Approach:** Explicitly trains models to predict the 5th percentile (lower bound) and 95th percentile (upper bound) to form a 90% prediction interval.

---

## 🛠 Installation & Requirements

To run the notebooks, you will need a Python environment with the following key libraries installed. You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost ngboost gpboost pgbm
pip install optuna optunahub hyperopt
pip install shap lime interpret
pip install mapie puncc
pip install plotly openpyxl

```

**Note:** Some notebooks (specifically those using `PGBM` or `NGBoost`) may require specific versions of `scikit-learn`. Check the first cell of the respective notebooks for version-specific installation commands.

---

## 🚀 Usage

1. **Clone the repository:**
```bash
git clone https://github.com/daneshselwal/concrete_data_analysis.git

```


2. **Navigate to the folder:**
```bash
cd concrete_data_analysis

```


3. **Run the notebooks:**
Start with `Hyperparameter_Tuning/Hyperparameter_tuning.ipynb` to see the baseline models, then proceed to `Model_Explainations` and `Uncertainity_Analysis` for deeper insights.
```bash
jupyter notebook

```



---

*This README was automatically generated to help you navigate the repository. For specific implementation details, please refer to the markdown cells within the individual Jupyter Notebooks.*
