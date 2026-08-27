# Heavy Equipment Selling Price Prediction Pipeline

An end-to-end Machine Learning pipeline designed to predict the auction selling prices of heavy industrial machinery (bulldozers, excavators, etc.) using regression modeling. 

This project explores data cleaning, leakage-free feature engineering, baseline model comparison, hyperparameter optimization, and high-capacity gradient boosting ensembles.

---

## 🚀 Project Overview

Predicting the market value of heavy machinery is a challenging tabular regression task due to high-cardinality categorical features, extreme outliers in operational usage, and non-linear depreciation curves. 

This pipeline processes historical transaction data of over **138,000 sales records** and trains a series of baseline and ensemble models to predict auction selling prices with high precision.

---

## 🛠️ Key Pipeline Features

1. **Exploratory Data Analysis (EDA):**
   * Visualized target variable skewness and applied log-transformations to normalize pricing distributions.
   * Analyzed outlier metrics using boxplots and identified default system placeholder codes (`1001` for year, `0.0` for hours) to replace them with null markers.

2. **Strict Data Leakage Prevention:**
   * Pre-split dataset indices into training and validation folds prior to calculations.
   * Computed IQR outlier clipping limits, temporal baseline reference dates, and category encoding mappings **strictly on training split rows** to prevent future information leaking into the model.

3. **Feature Engineering:**
   * Extracted temporal characteristics (Year, Month, Day of Week, Weekend indicator, elapsed time) from transaction date entries.
   * Engineered depreciation-related features including `MachineAge` (years since manufacture) and usage intensity (`HoursPerYear`).

4. **Category Encoding:**
   * Utilized scikit-learn's `OrdinalEncoder` to transform nominal features.
   * Configured the encoder to gracefully map unseen categories in the validation or test sets to a default value (`-1`) to prevent system crashes.

5. **Model Comparisons & Ensembles:**
   * Evaluated multiple regression models: **Ridge Regression**, **LinearSVR**, **Random Forest**, and **XGBoost**.
   * Built a **Stacking Regressor** meta-model combining linear and tree predictions.
   * Deployed a champion **LightGBM Regressor** to handle tabular patterns with high-cardinality categories.

6. **Optimized Hyperparameter Tuning:**
   * Implemented `RandomizedSearchCV` on a representative 5% subsample of the training split. This enabled rapid parameter search (completing in under 5 seconds) while preventing Kaggle timeout limits.

---

## 📈 Model Leaderboard (Validation Results)

| Model | $R^2$ Score | RMSLE (Root Mean Squared Log Error) |
|---|---|---|
| 🏆 **LightGBM (Champion)** | **0.9110** | **0.1982** (Fulfills `< 0.20` target) |
| Stacking Regressor | 0.8907 | 0.2157 |
| XGBoost Regressor | 0.8900 | 0.2160 |
| Random Forest | 0.8164 | 0.2637 |
| Ridge Regression | 0.4162 | 0.4874 |
| LinearSVR | -0.4165 | 0.7889 |

---

## 📁 Repository Structure

```text
├── 23f3004165-notebook-2026t2 (1).ipynb   # Cleaned, optimized project notebook
├── README.md                              # Project documentation and summary
└── .gitignore                             # Prevents massive datasets from uploading
```

---

## 💻 How to Run the Project

### Prerequisites
Install the required packages using pip:
```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
```

### Execution
1. Download the Heavy Equipment dataset from Kaggle.
2. Open Jupyter Notebook or Kaggle Notebook:
   ```bash
   jupyter notebook "23f3004165-notebook-2026t2 (1).ipynb"
   ```
3. Run all cells sequentially. The full execution completes in **under 4 minutes**.
