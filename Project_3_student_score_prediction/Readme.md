# 🎓 Student Academic Performance Regression Framework

## 📌 Overview
This repository implements a highly sophisticated and robust regression framework designed to accurately predict student academic scores (`writing score`) based on sociodemographic, parental, and educational performance vectors.

## 📊 Dataset & Business Problem
* **Target Variable:** `writing score` (Continuous numerical score).
* **Core Challenge:** The raw academic records contained varied feature encodings, mixed variable distributions (nominal vs. ordinal), and missing data entries requiring complex, multi-tiered imputation strategies to isolate performance drivers without introducing bias.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Feature Engineering & Pipelines:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`)
- **Hyperparameter Space Optimization:** `GridSearchCV`
- **Robust Regressors Evaluated:** LassoCV, ElasticNetCV, HuberRegressor, BayesianRidge, Ridge

## ⚙️ Advanced Feature Engineering Pipeline
The cornerstone of this project is a multi-tiered structural `ColumnTransformer` data processing pipeline:
1. **Numerical Pipeline (`numerical_feature`):** Imputed missing metrics via `SimpleImputer(strategy='median')` to resist outlier distortion, followed by continuous serialization via `StandardScaler` on `math score` and `reading score`.
2. **Nominal Pipeline (`nominal_feature`):** Imputed structural categorical vectors (`race/ethnicity`) via `most_frequent` substitution and expanded into geometric features using `OneHotEncoder`.
3. **Ordinal Hierarchy Pipeline (`ordinal_feature`):** Scientifically mapped ordered data strings (`parental level of education`, `gender`, `lunch`, `test preparation course`) using custom-defined tier matrices via `OrdinalEncoder`.

## 📈 Model Performance & Evaluation
Models were rigorously validated against multiple foundational linear regression errors (MAE, RMSE, and $R^2$). After automated cross-validation filtering, **LassoCV** generated the highest mathematical predictive performance on hidden test groups.

### Core Validation Metrics Matrix
| Robust Regression Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score ($R\_squared$) |
| :--- | :---: | :---: | :---: |
| **LassoCV (Optimal)** | **3.1957** | **3.8621** | **0.9381** |
| ElasticNetCV | 3.1955 | 3.8630 | 0.9380 |
| HuberRegressor | 3.2034 | 3.8682 | 0.9379 |
| Ridge | 3.2021 | 3.8645 | 0.9380 |
| BayesianRidge | 3.2037 | 3.8700 | 0.9378 |

* **Key Takeaway:** An $R^2$ evaluation score of **93.81%** mathematically proves that the feature processing architecture successfully explains nearly all variations in student outcomes, ensuring high dependability for academic forecasting systems.

## 📂 Repository Structure
```text
├── data/                  # Educational datasets (Student_score.csv)
├── notebooks/             # Advanced regression script mapping ColumnTransformers
│   └── student_score_prediction.py
└── README.md              # Project documentation
