# 🎮 CS:GO Match Outcome Prediction Classifier

## 📌 Overview
This repository contains an end-to-end Machine Learning classification framework designed to predict competitive Counter-Strike: Global Offensive (CS:GO) match outcomes (`Lost`, `Tie`, `Win`) based on real-time in-game performance metrics (such as kills, deaths, assists, mvps, and ping).

## 📊 Dataset & Business Problem
* **Dataset Source:** Historical CS:GO match performance records.
* **Target Variable:** `result` (Multi-class: `Lost`, `Tie`, `Win`).
* **Core Challenge:** Eliminating game metadata that triggers data leakage (e.g., specific round numbers per team) and engineering a parallel preprocessing pipeline that handles numeric statistics alongside non-linear game map factors seamlessly.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Auditing & Preprocessing:** Pandas, NumPy, Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
- **Automated Benchmarking:** `LazyPredict` (`LazyClassifier`)
- **Hyperparameter Optimization:** `GridSearchCV`
- **Primary Estimators Evaluated:** RandomForestClassifier, LogisticRegression, LinearSVC, ExtraTreesClassifier, CalibratedClassifierCV

## ⚙️ Engineering Workflow
1. **Feature Engineering & Leakage Prevention:** Dropped post-match summary columns (`team_a_rounds`, `team_b_rounds`) and temporal noise (`day`, `month`, `year`, `date`, time configurations) to evaluate pure in-game performance.
2. **Parallel Preprocessing Pipeline:** Implemented a robust `ColumnTransformer` to prevent data leakage during train-test splitting:
   - **Numerical (`numeric_feature`):** Scaled `ping`, `kills`, `assists`, `deaths`, `mvps`, `hs_percent`, `points` using `StandardScaler`.
   - **Categorical (`categories_feature`):** Encoded structural in-game maps (`map`) using `OneHotEncoder`.
3. **Model Benchmarking:** Leveraged `LazyClassifier` to efficiently evaluate baseline validation curves across dozens of estimators.
4. **Hyperparameter Tuning:** Conducted a rigorous multi-core grid search using 5-fold cross-validation (`GridSearchCV`) to fine-tune criterion limits, estimators, and class weights.

## 📈 Model Performance & Evaluation
After extensive hyperparameter tuning, **RandomForestClassifier** emerged as the optimal estimator, demonstrating powerful handling of non-linear tactical trends.

### Final Classification Report (RandomForestClassifier)
```text
              precision    recall  f1-score   support

        Lost       0.70      0.83      0.76       225
         Tie       0.33      0.05      0.09        37
         Win       0.78      0.73      0.76       192

    accuracy                           0.72       454
   macro avg       0.60      0.54      0.54       454
weighted avg       0.70      0.72      0.70       454
