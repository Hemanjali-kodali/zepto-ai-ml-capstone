import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "titanic.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        "analytics/titanic.csv not found. Run 01_eda.py first."
    )

df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

cleaned_df = df.copy()

if "deck" in cleaned_df.columns:
    cleaned_df = cleaned_df.drop(columns=["deck"])

cleaned_df = cleaned_df.dropna(
    subset=["embarked", "embark_town"]
)

cleaned_df["age"] = cleaned_df["age"].fillna(
    cleaned_df["age"].median()
)

if "alive" in cleaned_df.columns:
    cleaned_df = cleaned_df.drop(columns=["alive"])

target = "survived"

X = cleaned_df.drop(columns=[target])
y = cleaned_df[target]

numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked",
    "class",
    "who",
    "adult_male",
    "embark_town",
    "alone"
]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

print("\n===== PREPROCESSING COMPLETE =====")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

classification_results = []


def evaluate_classifier(name, model):

    model.fit(
        X_train_processed,
        y_train
    )

    predictions = model.predict(
        X_test_processed
    )

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    cm = confusion_matrix(
        y_test,
        predictions
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n=====", name, "=====")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", auc)
    print("Confusion Matrix:")
    print(cm)

    classification_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })

    return model, probabilities, cm


logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model, logistic_prob, logistic_cm = evaluate_classifier(
    "Logistic Regression",
    logistic_model
)

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model, tree_prob, tree_cm = evaluate_classifier(
    "Decision Tree",
    tree_model
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model, rf_prob, rf_cm = evaluate_classifier(
    "Random Forest",
    rf_model
)

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    max_depth=4
)

plt.title("Decision Tree")
plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "decision_tree.png"
    ),
    dpi=150
)

plt.show()

for matrix, title in [
    (logistic_cm, "Logistic Regression Confusion Matrix"),
    (tree_cm, "Decision Tree Confusion Matrix"),
    (rf_cm, "Random Forest Confusion Matrix")
]:

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d"
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_prob
)

tree_fpr, tree_tpr, _ = roc_curve(
    y_test,
    tree_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label="Logistic Regression"
)

plt.plot(
    tree_fpr,
    tree_tpr,
    label="Decision Tree"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label="Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "roc_curves.png"
    ),
    dpi=150
)

plt.show()

print("\n===== CLASS BALANCE =====")

class_balance = y.value_counts()

print(class_balance)

print("\nClass percentages:")

print(
    (y.value_counts(normalize=True) * 100).round(2)
)

imbalance_results = []


def evaluate_imbalance(
    name,
    model,
    train_X,
    train_y
):

    model.fit(
        train_X,
        train_y
    )

    predictions = model.predict(
        X_test_processed
    )

    imbalance_results.append({
        "Strategy": name,
        "Precision": precision_score(
            y_test,
            predictions
        ),
        "Recall": recall_score(
            y_test,
            predictions
        ),
        "F1": f1_score(
            y_test,
            predictions
        )
    })


baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

evaluate_imbalance(
    "Baseline",
    baseline_model,
    X_train_processed,
    y_train
)

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

evaluate_imbalance(
    "Class Weight Balanced",
    balanced_model,
    X_train_processed,
    y_train
)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\n===== SMOTE =====")
print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

smote_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

evaluate_imbalance(
    "SMOTE",
    smote_model,
    X_train_smote,
    y_train_smote
)

imbalance_results_df = pd.DataFrame(
    imbalance_results
)

print("\n===== IMBALANCE COMPARISON =====")

print(
    imbalance_results_df.round(4).to_string(
        index=False
    )
)

rf_grid_model = RandomForestClassifier(
    random_state=42,
    oob_score=True
)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=rf_grid_model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_search.fit(
    X_train_processed,
    y_train
)

best_rf = grid_search.best_estimator_

print("\n===== RANDOM FOREST GRID SEARCH =====")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1:")
print(grid_search.best_score_)

print("\nOOB Score:")
print(best_rf.oob_score_)

tuned_rf_predictions = best_rf.predict(
    X_test_processed
)

tuned_rf_probabilities = best_rf.predict_proba(
    X_test_processed
)[:, 1]

tuned_rf_cm = confusion_matrix(
    y_test,
    tuned_rf_predictions
)

tuned_rf_accuracy = accuracy_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_precision = precision_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_recall = recall_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_f1 = f1_score(
    y_test,
    tuned_rf_predictions
)

tuned_rf_auc = roc_auc_score(
    y_test,
    tuned_rf_probabilities
)

classification_results.append({
    "Model": "Tuned Random Forest",
    "Accuracy": tuned_rf_accuracy,
    "Precision": tuned_rf_precision,
    "Recall": tuned_rf_recall,
    "F1": tuned_rf_f1,
    "ROC-AUC": tuned_rf_auc
})

print("\n===== TUNED RANDOM FOREST =====")
print("Accuracy :", tuned_rf_accuracy)
print("Precision:", tuned_rf_precision)
print("Recall   :", tuned_rf_recall)
print("F1 Score :", tuned_rf_f1)
print("ROC-AUC  :", tuned_rf_auc)
print("Confusion Matrix:")
print(tuned_rf_cm)

regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

X_reg = cleaned_df[
    regression_features
]

y_reg = cleaned_df[
    "fare"
]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

regression_model = LinearRegression()

regression_model.fit(
    X_reg_train,
    y_reg_train
)

reg_predictions = regression_model.predict(
    X_reg_test
)

reg_mae = mean_absolute_error(
    y_reg_test,
    reg_predictions
)

reg_mse = mean_squared_error(
    y_reg_test,
    reg_predictions
)

reg_rmse = np.sqrt(
    reg_mse
)

reg_r2 = r2_score(
    y_reg_test,
    reg_predictions
)

n = len(y_reg_test)
p = X_reg_test.shape[1]

reg_adjusted_r2 = (
    1
    - (
        (1 - reg_r2)
        * (n - 1)
        / (n - p - 1)
    )
)

print("\n===== REGRESSION SIDE TASK =====")
print("MAE        :", reg_mae)
print("RMSE       :", reg_rmse)
print("R2         :", reg_r2)
print("Adjusted R2:", reg_adjusted_r2)

residuals = (
    y_reg_test
    - reg_predictions
)

plt.figure(figsize=(8, 5))

plt.scatter(
    reg_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title(
    "Residual Plot - Multivariate Linear Regression"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "linear_regression_residuals.png"
    ),
    dpi=150
)

plt.show()

absolute_residuals = np.abs(
    residuals
)

spread_correlation = np.corrcoef(
    reg_predictions,
    absolute_residuals
)[0, 1]

print(
    "\nAbsolute residual / prediction correlation:"
)

print(
    spread_correlation
)

if abs(spread_correlation) >= 0.30:

    heteroscedasticity_conclusion = (
        "The residual plot shows evidence of "
        "heteroscedasticity because the spread of "
        "residuals changes systematically as "
        "predicted fare increases."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong "
        "evidence of heteroscedasticity. The "
        "residuals are reasonably randomly distributed "
        "around zero."
    )

print(
    "\nHeteroscedasticity conclusion:"
)

print(
    heteroscedasticity_conclusion
)

classification_results_df = pd.DataFrame(
    classification_results
)

regression_results_df = pd.DataFrame({
    "Regression Model": [
        "Multivariate Linear Regression"
    ],
    "MAE": [
        reg_mae
    ],
    "RMSE": [
        reg_rmse
    ],
    "R2": [
        reg_r2
    ],
    "Adjusted R2": [
        reg_adjusted_r2
    ]
})

print("\n===== CLASSIFICATION MODEL COMPARISON =====")

print(
    classification_results_df.round(4).to_string(
        index=False
    )
)

print("\n===== REGRESSION MODEL COMPARISON =====")

print(
    regression_results_df.round(4).to_string(
        index=False
    )
)

print("\n===== COMPLETE MODEL COMPARISON =====")

complete_comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Model Type": "Classification",
        "Accuracy": classification_results_df.loc[
            classification_results_df["Model"] == "Logistic Regression",
            "Accuracy"
        ].iloc[0],
        "Precision": classification_results_df.loc[
            classification_results_df["Model"] == "Logistic Regression",
            "Precision"
        ].iloc[0],
        "Recall": classification_results_df.loc[
            classification_results_df["Model"] == "Logistic Regression",
            "Recall"
        ].iloc[0],
        "F1": classification_results_df.loc[
            classification_results_df["Model"] == "Logistic Regression",
            "F1"
        ].iloc[0],
        "ROC-AUC": classification_results_df.loc[
            classification_results_df["Model"] == "Logistic Regression",
            "ROC-AUC"
        ].iloc[0],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Adjusted R2": np.nan
    },
    {
        "Model": "Decision Tree",
        "Model Type": "Classification",
        "Accuracy": classification_results_df.loc[
            classification_results_df["Model"] == "Decision Tree",
            "Accuracy"
        ].iloc[0],
        "Precision": classification_results_df.loc[
            classification_results_df["Model"] == "Decision Tree",
            "Precision"
        ].iloc[0],
        "Recall": classification_results_df.loc[
            classification_results_df["Model"] == "Decision Tree",
            "Recall"
        ].iloc[0],
        "F1": classification_results_df.loc[
            classification_results_df["Model"] == "Decision Tree",
            "F1"
        ].iloc[0],
        "ROC-AUC": classification_results_df.loc[
            classification_results_df["Model"] == "Decision Tree",
            "ROC-AUC"
        ].iloc[0],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Adjusted R2": np.nan
    },
    {
        "Model": "Random Forest",
        "Model Type": "Classification",
        "Accuracy": classification_results_df.loc[
            classification_results_df["Model"] == "Random Forest",
            "Accuracy"
        ].iloc[0],
        "Precision": classification_results_df.loc[
            classification_results_df["Model"] == "Random Forest",
            "Precision"
        ].iloc[0],
        "Recall": classification_results_df.loc[
            classification_results_df["Model"] == "Random Forest",
            "Recall"
        ].iloc[0],
        "F1": classification_results_df.loc[
            classification_results_df["Model"] == "Random Forest",
            "F1"
        ].iloc[0],
        "ROC-AUC": classification_results_df.loc[
            classification_results_df["Model"] == "Random Forest",
            "ROC-AUC"
        ].iloc[0],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Adjusted R2": np.nan
    },
    {
        "Model": "Tuned Random Forest",
        "Model Type": "Classification",
        "Accuracy": tuned_rf_accuracy,
        "Precision": tuned_rf_precision,
        "Recall": tuned_rf_recall,
        "F1": tuned_rf_f1,
        "ROC-AUC": tuned_rf_auc,
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Adjusted R2": np.nan
    },
    {
        "Model": "Multivariate Linear Regression",
        "Model Type": "Regression",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "ROC-AUC": np.nan,
        "MAE": reg_mae,
        "RMSE": reg_rmse,
        "R2": reg_r2,
        "Adjusted R2": reg_adjusted_r2
    }
])

print(
    complete_comparison.round(4).to_string(
        index=False
    )
)

classification_results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "classification_model_comparison.csv"
    ),
    index=False
)

imbalance_results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "imbalance_comparison.csv"
    ),
    index=False
)

regression_results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "regression_results.csv"
    ),
    index=False
)

complete_comparison.to_csv(
    os.path.join(
        BASE_DIR,
        "complete_model_comparison.csv"
    ),
    index=False
)

best_f1_row = classification_results_df.loc[
    classification_results_df["F1"].idxmax()
]

print("\n===== FINAL RECOMMENDATION =====")

print(
    f"{best_f1_row['Model']} achieved the highest F1 "
    f"score of {best_f1_row['F1']:.4f} among the "
    f"evaluated final classifiers."
)

print(
    f"It achieved {best_f1_row['Accuracy']:.4f} accuracy "
    f"and {best_f1_row['ROC-AUC']:.4f} ROC-AUC."
)

print(
    f"Tuned Random Forest achieved the highest precision "
    f"of {tuned_rf_precision:.4f}, while Balanced Logistic "
    f"Regression achieved recall of 0.7647."
)

print(
    "Based on the combined F1 and ROC-AUC results, "
    "Logistic Regression was selected as the final "
    "classifier for the saved pipeline."
)

if best_f1_row["Model"] == "Logistic Regression":

    final_estimator = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

elif best_f1_row["Model"] == "Decision Tree":

    final_estimator = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

elif best_f1_row["Model"] == "Random Forest":

    final_estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

else:

    final_estimator = best_rf

full_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        final_estimator
    )
])

full_pipeline.fit(
    X_train,
    y_train
)

pipeline_path = os.path.join(
    BASE_DIR,
    "titanic_model_pipeline.joblib"
)

joblib.dump(
    full_pipeline,
    pipeline_path
)

print("\n===== PIPELINE SAVED =====")

print(
    "Saved to:",
    pipeline_path
)

loaded_pipeline = joblib.load(
    pipeline_path
)

sample_raw_input = X_test.iloc[
    [0]
]

loaded_prediction = loaded_pipeline.predict(
    sample_raw_input
)

print("\n===== RELOAD TEST =====")

print("Raw input:")
print(sample_raw_input)

print("\nPredicted value:")
print(loaded_prediction)

print("\nActual value:")
print(
    y_test.iloc[0]
)

print(
    "\nReload and prediction test successful."
)

print(
    "\n===== MODULE 2 COMPLETED ====="
)