# ============================================================
# LOAN APPROVAL PREDICTION
# Logistic Regression + Decision Tree
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_FILE = "loan_approval_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 60)
print("LOAN APPROVAL PREDICTION - MODEL TRAINING")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# ============================================================
# 2. VERIFY DATASET
# ============================================================

print("\n" + "=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nTarget Distribution:")
print(df["loan_status"].value_counts())

print("\nTarget Distribution Percentage:")
print(
    df["loan_status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop("loan_status", axis=1)

y = df["loan_status"]

print("\nFeature Shape:", X.shape)
print("Target Shape :", y.shape)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

logistic_model.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed.")


# ============================================================
# 6. DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DECISION TREE")
print("=" * 60)

decision_tree_model = DecisionTreeClassifier(
    max_depth=2,
    min_samples_leaf=10,
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

print("Decision Tree training completed.")


# ============================================================
# 7. PREDICTIONS
# ============================================================

logistic_predictions = logistic_model.predict(
    X_test
)

tree_predictions = decision_tree_model.predict(
    X_test
)


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    model,
    X_test
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Rejected",
                "Approved"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_true,
        predictions
    )

    print(cm)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }


# ============================================================
# 9. EVALUATE LOGISTIC REGRESSION
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_model,
    X_test
)


# ============================================================
# 10. EVALUATE DECISION TREE
# ============================================================

tree_results = evaluate_model(
    "Decision Tree",
    y_test,
    tree_predictions,
    decision_tree_model,
    X_test
)


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    logistic_results,
    tree_results
])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)


# ============================================================
# 12. PRINT CONFUSION MATRICES CLEARLY
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)

print("\nRows = Actual")
print("Columns = Predicted")
print("Order = [Rejected, Approved]")


print("\n" + "=" * 60)
print("DECISION TREE CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        tree_predictions
    )
)

print("\nRows = Actual")
print("Columns = Predicted")
print("Order = [Rejected, Approved]")


# ============================================================
# 13. DECISION TREE FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": decision_tree_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("DECISION TREE FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 14. DISPLAY DECISION TREE
# ============================================================

plt.figure(
    figsize=(18, 9)
)

plot_tree(
    decision_tree_model,
    feature_names=X.columns,
    class_names=[
        "Rejected",
        "Approved"
    ],
    filled=True,
    rounded=True
)

plt.title(
    "Decision Tree for Loan Approval Prediction"
)

plt.tight_layout()

plt.show()


# ============================================================
# 15. SAVE MODELS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    logistic_model,
    "models/logistic_model.pkl"
)

joblib.dump(
    decision_tree_model,
    "models/decision_tree_model.pkl"
)

print("\n" + "=" * 60)
print("MODELS SAVED SUCCESSFULLY")
print("=" * 60)

print(
    "Logistic Regression:"
    " models/logistic_model.pkl"
)

print(
    "Decision Tree:"
    " models/decision_tree_model.pkl"
)


# ============================================================
# 16. TEST SAMPLE APPLICANT
# ============================================================

sample_applicant = pd.DataFrame({

    "gender": [1],

    "married": [1],

    "dependents": [0],

    "education": [1],

    "self_employed": [0],

    "applicantincome": [5000],

    "coapplicantincome": [2000],

    "loanamount": [150],

    "loan_amount_term": [360],

    "credit_history": [1],

    "property_area": [2]
})


# ============================================================
# 17. SAMPLE PREDICTIONS
# ============================================================

logistic_result = logistic_model.predict(
    sample_applicant
)[0]

logistic_probability = logistic_model.predict_proba(
    sample_applicant
)[0][1]


tree_result = decision_tree_model.predict(
    sample_applicant
)[0]

tree_probability = decision_tree_model.predict_proba(
    sample_applicant
)[0][1]


print("\n" + "=" * 60)
print("SAMPLE APPLICANT PREDICTION")
print("=" * 60)


print("\nInput Applicant:")
print(sample_applicant.to_string(index=False))


print("\nLogistic Regression:")

if logistic_result == 1:
    print("Prediction : LOAN APPROVED")
else:
    print("Prediction : LOAN REJECTED")

print(
    f"Approval Probability : "
    f"{logistic_probability * 100:.2f}%"
)


print("\nDecision Tree:")

if tree_result == 1:
    print("Prediction : LOAN APPROVED")
else:
    print("Prediction : LOAN REJECTED")

print(
    f"Approval Probability : "
    f"{tree_probability * 100:.2f}%"
)


print("\nModel Agreement:")

if logistic_result == tree_result:

    print(
        "Both models give the SAME prediction."
    )

else:

    print(
        "The models give DIFFERENT predictions."
    )


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING AND TESTING COMPLETED SUCCESSFULLY")
print("=" * 60)