# 🧠 Stroke Risk Prediction Pipeline with Imbalance Mitigation

## 📌 Overview
This project delivers a machine learning pipeline designed to assess patient stroke risks based on multi-dimensional clinical and demographic variables. The framework specifically handles the critical healthcare challenge of severe class imbalance to construct a reliable early warning system.

## 📊 Dataset & Business Problem
* **Target Variable:** `stroke` (0: Healthy, 1: High Stroke Risk).
* **Primary Challenge:** Extreme class imbalance (`stroke=1` represents a tiny fraction of total patient records). Standard training without adjustments yields models that score high overall accuracy but completely fail to recall actual stroke patients, which is life-threatening in medical deployments.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Cleansing & Profiling:** Pandas, NumPy, `ydata-profiling`
- **Preprocessing & Pipelines:** Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
- **Imbalance Handling:** Imbalanced-Learn (`SMOTE`)
- **Hyperparameter Exploration:** `LazyPredict`, `GridSearchCV`
- **Core Models Tested:** BaggingClassifier, NearestCentroid, ExtraTreesClassifier, RandomForestClassifier, KNeighborsClassifier

## ⚙️ Engineering Workflow
1. **Clinical Data Sanitization:** Conducted explicit row-wise drops of incomplete records and removed arbitrary tracking tokens (`pat_id`).
2. **Modular Preprocessing:** Combined structural transformations via Scikit-Learn pipelines:
   - **Numerical Constraints:** Standardized `age`, `bmi`, `avg_glucose_level`, smoke histories, and stress vectors using `StandardScaler`.
   - **Categorical Constraints:** Transformed clinical features (`gender`) using `OneHotEncoder`.
3. **Data Leakage Immunization:** Embedded Synthetic Minority Over-sampling Technique (`SMOTE`) directly into an **`imb_pipeline` (Imbalanced-Learn)** workflow. This ensures over-sampling happens strictly within cross-validation folds, entirely removing data leakage.
4. **Optimization Tuning:** Benchmarked with `LazyClassifier` and performed rigorous optimization with `GridSearchCV` optimizing for the `f1_macro` scoring metric to focus heavily on the minority class.

## 📈 Model Performance & Evaluation
Evaluating multiple tuned pipelines highlighted important predictive characteristics across classifiers:

### 1. Robust Ensemble Approach: `BaggingClassifier`
* Highly stable and excellent overall generalization.
* **Overall Accuracy:** **92%**
```text
              precision    recall  f1-score   support
           0       0.95      0.97      0.96      1401
           1       0.14      0.11      0.12        72
