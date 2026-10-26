# Module 2 — Analytics & Machine Learning: End-to-End Predictive Modeling Pipeline

This module represents the complete analyst-to-data-scientist workflow in one cohesive pass: profiling raw data, applying threshold-based data cleaning, crafting a visual data story, and building a leak-free predictive modeling pipeline with hyperparameter tuning, imbalance handling, regression benchmarking, and model serialization.

---

## Workflow Architecture

```
Titanic Dataset (sns.load_dataset('titanic') / offline fallback)
                       │
                       ▼
              analytics/titanic.csv (Committed Offline Fallback)
                       │
                       ▼
             01_eda.py (Profiling, Cleaning, Visual Data Story)
           ┌───────────┴───────────┐
           ▼                       ▼
Threshold Cleaning (889 rows)    Exploratory Z-Score Sanity Check
           │
           ▼
     02_modeling.py (Predictive Modeling & Regression)
           │
           ├─► Stratified Train/Test Split (80/20, Stratify=y)
           │
           ├─► Preprocessing Pipeline (Fit on Train Fold ONLY)
           │     ├── Numeric: SimpleImputer(median) + StandardScaler
           │     └── Categorical: SimpleImputer(most_frequent) + OneHotEncoder
           │
           ├─► Classification Models (Identical Split):
           │     ├── Logistic Regression
           │     ├── Decision Tree (plot_tree in decision_tree.png)
           │     ├── Random Forest (ROC curves in roc_curves.png)
           │     └── Tuned Random Forest (GridSearchCV + OOB score)
           │
           ├─► Class Imbalance Experiment (Baseline vs. Balanced vs. SMOTE)
           │
           ├─► Regression Task (Predicting Fare: MAE, RMSE, R², Adj R², Residuals)
           │
           └─► Production Serialization (joblib.dump(full_pipeline))
                 └── Reload & verify on raw unpreprocessed input
```

---

## Part A — Profiling, Cleaning, and the Data Story

### 1. Dataset Profiling & Single Offline Fallback
- **Loading Strategy**: Loaded once via `sns.load_dataset("titanic")` and saved immediately as [`analytics/titanic.csv`](file:///home/murali/Downloads/zepto-ai-ml-capstone/analytics/titanic.csv). If network access is unavailable, [`01_eda.py`](file:///home/murali/Downloads/zepto-ai-ml-capstone/analytics/01_eda.py) and [`02_modeling.py`](file:///home/murali/Downloads/zepto-ai-ml-capstone/analytics/02_modeling.py) automatically load this committed CSV.
- **Original Dimensions**: 891 rows, 15 columns.

### 2. Missing-Value Handling per Threshold Rule
Every column with missing values was evaluated against the project's explicit percentage thresholds:

| Column | Missing Count | Missing % | Applied Strategy | Threshold Rule Justification |
| :--- | :---: | :---: | :--- | :--- |
| **`age`** | 177 | 19.87% | **Median Imputation** | Falls within the 5%–30% threshold. Median imputation preserves sample size and central tendency without being distorted by age outliers. |
| **`embarked`** | 2 | 0.22% | **Drop Rows** | Below the 5% threshold (< 0.25%). Dropping these 2 rows has virtually zero impact on statistical power. |
| **`embark_town`**| 2 | 0.22% | **Drop Rows** | Below the 5% threshold (corresponds to the same 2 passengers as `embarked`). |
| **`deck`** | 688 | 77.22% | **Drop Column** | Far exceeds the 30% threshold (> 77% missing). Imputing over three-quarters of missing values would introduce fabricated data and severe bias. |

**Cleaned Dataset**: 889 rows, 14 columns, 0 missing values.

### 3. Univariate & Skewness Analysis
- **IQR Outlier Counts**:
  - `age`: **65 outliers** ($Q1 = 22.0, Q3 = 35.0, \text{IQR} = 13.0$, Bounds: $[2.5, 54.5]$).
  - `fare`: **114 outliers** ($Q1 = 7.90, Q3 = 31.0, \text{IQR} = 23.10$, Bounds: $[-26.76, 65.66]$).
- **Fare Distribution Skewness**:
  - $\text{Mode} (8.05) < \text{Median} (14.45) < \text{Mean} (32.10)$
  - $\text{Skewness} = +4.8014$
  - *Written Conclusion*: The ordering $\text{Mode} < \text{Median} < \text{Mean}$ combined with large positive skewness demonstrates that ticket fares are **strongly right-skewed**, characterized by a dense concentration of economy-class fares under £15 and a long right tail of luxury tickets up to £512.

### 4. Bivariate Analysis (via Boolean Masking)
Calculated using explicit boolean masks with `&` combinations:
- **(a) By Sex**:
  - Female: **74.04%** (`df[df['sex'] == 'female']['survived'].mean() * 100`)
  - Male: **18.89%** (`df[df['sex'] == 'male']['survived'].mean() * 100`)
- **(b) By Passenger Class (`pclass`)**:
  - Class 1: **62.62%**
  - Class 2: **47.28%**
  - Class 3: **24.24%**
- **(c) By Sex and Class Combined (`&` combinations)**:
  - Female, Class 1: **96.74%**
  - Female, Class 2: **92.11%**
  - Female, Class 3: **50.00%**
  - Male, Class 1: **36.89%**
  - Male, Class 2: **15.74%**
  - Male, Class 3: **13.54%**

### 5. 6×6 Correlation Matrix (Numeric Only)
Restricted strictly to the 6 numeric features (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`), explicitly omitting derived redundant boolean flags `adult_male` and `alone`:

| Feature | survived | pclass | age | sibsp | parch | fare |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **survived** | 1.0000 | -0.3355 | -0.0698 | -0.0340 | 0.0832 | 0.2553 |
| **pclass**   | -0.3355 | 1.0000 | -0.3365 | 0.0817 | 0.0168 | -0.5482 |
| **age**      | -0.0698 | -0.3365 | 1.0000 | -0.2325 | -0.1715 | 0.0937 |
| **sibsp**    | -0.0340 | 0.0817 | -0.2325 | 1.0000 | 0.4145 | 0.1609 |
| **parch**    | 0.0832 | 0.0168 | -0.1715 | 0.4145 | 1.0000 | 0.2175 |
| **fare**     | 0.2553 | -0.5482 | 0.0937 | 0.1609 | 0.2175 | 1.0000 |

**Interpretation of Top Two Off-Diagonal Correlations**:
1. **`pclass` vs. `fare` ($r = -0.5482$, $|r| = 0.5482$)**: The strongest correlation in the dataset. This negative relationship confirms that higher ticket fares correspond directly to lower numeric class values (Class 1 vs. Class 3).
2. **`sibsp` vs. `parch` ($r = 0.4145$, $|r| = 0.4145$)**: Moderate positive correlation demonstrating that passengers traveling with siblings or spouses were substantially more likely to travel with parents or children as part of coordinated family units.

### 6. Multivariate Data Story (4 Distinct Charts & Interpretations)
- **Chart 1 — Survival Rate by Sex**: Female passengers exhibited a 74.04% survival rate compared to only 18.89% for males. This striking disparity reflects the strict adherence to the maritime "women and children first" evacuation protocol during lifeboat boarding.
- **Chart 2 — Survival Rate by Passenger Class**: First-class passengers survived at 62.62%, second-class at 47.28%, and third-class at 24.24%. Physical proximity to the boat deck and socioeconomic prioritization gave higher-class passengers an enormous survival advantage.
- **Chart 3 — Survival Rate by Sex and Passenger Class Combined**: First and second-class females achieved near-universal survival rates of 96.74% and 92.11% respectively, whereas third-class females had an even 50.00% survival rate. Third-class males experienced the lowest survival rate of any cohort at 13.54%, illustrating the compounded effect of gender and class hierarchy.
- **Chart 4 — Age vs. Fare by Survival Status**: Passengers who paid fares exceeding £100 survived in overwhelming proportions regardless of age. Non-survivors are overwhelmingly clustered at low ticket prices (under £30), demonstrating that financial investment in cabin location was one of the strongest survival determinants.

### 7. Exploratory Standardization Sanity Check
Applied on full cleaned data for exploratory verification:
- **Before Standardization**:
  - `age`: $\text{Mean} = 29.64, \text{Std} = 13.01$
  - `fare`: $\text{Mean} = 32.10, \text{Std} = 49.70$
- **After Z-Score Standardization ($z = \frac{x - \mu}{\sigma}$)**:
  - `age_scaled`: $\text{Mean} = 0.00, \text{Std} = 1.00$
  - `fare_scaled`: $\text{Mean} = 0.00, \text{Std} = 1.00$

---

## Part B — Predictive Modeling Pipeline

### 8. Stratified Train/Test Split
- **Split Ratio**: 80% Train ($N=711$), 20% Test ($N=178$), `random_state=42`.
- **Target**: `survived`.
- **Stratification Justification**: The target exhibits a 61.62% / 38.38% class imbalance (549 non-survivors vs. 342 survivors). A simple random split risks sampling an unrepresentative class proportion in the test split, distorting precision and recall. Stratification guarantees that the 61.6% : 38.4% ratio is preserved exactly across both training and testing folds.

### 9. Leak-Free Preprocessing Pipeline
Implemented using `scikit-learn` `ColumnTransformer` and fit **strictly on `X_train` only**, applied in transform-only mode to `X_test`:
- **Numeric Features (`age`, `sibsp`, `parch`, `fare`)**: `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`.
- **Categorical Features (`sex`, `embarked`, `class`, `who`, `adult_male`, `embark_town`, `alone`)**: `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Total encoded feature dimensions: **22 processed columns**.

### 10. Classification Model Comparison
All models trained on the identical stratified training split:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8146** | 0.7966 | **0.6912** | **0.7402** | **0.8680** |
| **Decision Tree** | 0.7921 | 0.8163 | 0.5882 | 0.6838 | 0.8248 |
| **Random Forest** | 0.7809 | 0.7302 | 0.6765 | 0.7023 | 0.8155 |
| **Tuned Random Forest** | **0.8146** | **0.8182** | 0.6618 | 0.7317 | 0.8336 |

- **Decision Tree Visualization**: Saved as [`analytics/decision_tree.png`](file:///home/murali/Downloads/zepto-ai-ml-capstone/analytics/decision_tree.png) with labeled features and classes (`Not Survived` / `Survived`).
- **ROC Curves**: Saved as [`analytics/roc_curves.png`](file:///home/murali/Downloads/zepto-ai-ml-capstone/analytics/roc_curves.png).

### 11. Class Imbalance Handling Comparison
Evaluated on Logistic Regression across three strategies:

| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **(a) Baseline** | **0.7966** | 0.6912 | 0.7402 |
| **(b) Class Weight Balanced** | 0.7879 | **0.7647** | **0.7761** |
| **(c) SMOTE (Train Fold Only)** | 0.7937 | 0.7353 | 0.7634 |

- *Written Conclusion*: The **`class_weight='balanced'`** strategy proved most effective, boosting minority-class recall from 0.6912 to 0.7647 and achieving the highest F1 score of 0.7761. Penalizing errors proportionally in the loss function effectively balanced false negatives without synthesizing artificial points. SMOTE also improved recall (0.7353) without data leakage, but balanced weights offered superior performance.

### 12. Hyperparameter Tuning (Random Forest)
- Tuned using `GridSearchCV(cv=5, scoring='f1', n_jobs=-1)` on `RandomForestClassifier(oob_score=True, random_state=42)`.
- **Search Space**: `n_estimators`: `[50, 100]`, `max_depth`: `[None, 5, 10]`, `max_features`: `['sqrt', 'log2']`.
- **Best Parameters**: `{'n_estimators': 100, 'max_depth': 5, 'max_features': 'sqrt'}`
- **Best Cross-Validation F1**: **0.7542**
- **Out-of-Bag (OOB) Score**: **0.8228**

### 13. Regression Side-Task (Predicting `fare`)
Multivariate Linear Regression trained using predictors `pclass`, `age`, `sibsp`, and `parch`:
- **MAE**: **20.6867**
- **RMSE**: **42.5706**
- **$R^2$**: **0.3207**
- **Adjusted $R^2$**: **0.3050**
- **Heteroscedasticity Analysis**:
  - Saved residual plot: [`analytics/linear_regression_residuals.png`](file:///home/murali/Downloads/zepto-ai-ml-capstone/analytics/linear_regression_residuals.png)
  - Correlation between absolute residuals and predicted values: **$r = 0.3651$**
  - *Written Conclusion*: The residual plot exhibits clear **heteroscedasticity**. Residual variance fans outward as predicted fare increases, reflecting that ticket prices for economy passengers have very small variance while luxury tickets have large pricing swings that cannot be captured by linear predictors alone.

---

## Complete Model Comparison Table

Classification and regression metrics evaluate fundamentally different tasks and are presented in distinct metric columns:

| Model | Task / Type | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE | RMSE | R² | Adj R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Classification | 0.8146 | 0.7966 | 0.6912 | **0.7402** | **0.8680** | — | — | — | — |
| **Decision Tree** | Classification | 0.7921 | 0.8163 | 0.5882 | 0.6838 | 0.8248 | — | — | — | — |
| **Random Forest** | Classification | 0.7809 | 0.7302 | 0.6765 | 0.7023 | 0.8155 | — | — | — | — |
| **Tuned Random Forest** | Classification | 0.8146 | **0.8182** | 0.6618 | 0.7317 | 0.8336 | — | — | — | — |
| **Linear Regression** | Regression | — | — | — | — | — | 20.6867 | 42.5706 | 0.3207 | 0.3050 |

### Final Deployment Recommendation
For production deployment, **Logistic Regression** is the recommended classifier. It achieved the highest overall test F1 score (**0.7402**) and the highest ROC-AUC (**0.8680**) among the baseline classifiers, while matching Tuned Random Forest's top accuracy (**0.8146**). In addition to superior discrimination ability across probability thresholds, Logistic Regression offers fast sub-millisecond inference latency, minimal memory overhead, and clear coefficient interpretability for business analysts. Where false negative minimization is paramount, applying balanced class weighting further boosts recall to **0.7647**.

---

## Complete Saved Pipeline Artifact

- **File**: [`analytics/titanic_model_pipeline.joblib`](file:///home/murali/Downloads/zepto-ai-ml-capstone/analytics/titanic_model_pipeline.joblib)
- **Contents**: Complete composite `sklearn.pipeline.Pipeline` object encapsulating `ColumnTransformer` (median imputers, standard scaler, one-hot encoder) coupled with the fitted estimator.
- **Reload Verification**: Reloaded via `joblib.load()` and evaluated on raw unpreprocessed DataFrame rows, returning successful predictions without external preprocessing scripts.

---

## Reproduction Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Exploratory Data Analysis & visual story
python analytics/01_eda.py

# 3. Run predictive modeling, tuning, regression & pipeline serialization
python analytics/02_modeling.py
```
