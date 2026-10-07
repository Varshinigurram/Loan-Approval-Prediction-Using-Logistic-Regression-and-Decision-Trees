# ============================================================
# LOAN APPROVAL PREDICTION
# Streamlit Application
# Logistic Regression + Decision Tree
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- General page ---------- */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 750;
        text-align: center;
        margin: 0;
        letter-spacing: -0.8px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #667085;
        margin-top: 6px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin: 10px 0 16px 0;
    }

    /* ---------- Input area ---------- */
    .input-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 20px 22px 10px 22px;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
    }

    .input-card-title {
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* ---------- Prediction result cards ---------- */
    .prediction-card {
        border-radius: 18px;
        padding: 24px;
        min-height: 205px;
        border: 1px solid #e4e7ec;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.06);
    }

    .prediction-card.approved {
        background: #f0fdf4;
        border-color: #bbf7d0;
    }

    .prediction-card.rejected {
        background: #fef2f2;
        border-color: #fecaca;
    }

    .model-name {
        font-size: 20px;
        font-weight: 650;
        color: #751515;
        margin-bottom: 14px;
    }

    .prediction-status {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .approved-text {
        color: #15803d;
    }

    .rejected-text {
        color: #b42318;
    }

    .probability-label {
        color: #667085;
        font-size: 14px;
        margin-bottom: 3px;
    }

    .probability-value {
        font-size: 29px;
        font-weight: 800;
        color: #101828;
    }

    .probability-track {
        width: 100%;
        height: 8px;
        background: #e4e7ec;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 10px;
    }

    .probability-fill {
        height: 100%;
        border-radius: 20px;
        background: #1570ef;
    }

    /* ---------- Comparison card ---------- */
    .comparison-card {
        margin-top: 22px;
        border: 3px solid #e4e7ed;
        border-radius: 16px;
        padding: 18px 22px;
        background: #f5eeed;
        text-align: center;
    }

    .comparison-title {
        font-size: 15px;
        font-weight: 700;
        color: #171410;
        margin-bottom: 5px;
    }

    .comparison-result {
        font-size: 18px;
        font-weight: 550;
        color: #171410;
    }

    .comparison-note {
        font-size: 15px;
        color: #993314;
        margin-top: 5px;
    }

    /* ---------- Summary card ---------- */
    .summary-card {
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 20px 24px;
        background: #f5e3fc;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
    }

    .summary-row {
        padding: 8px 0;
        border-bottom: 1px solid #1f1c1c;
        font-size: 15px;
    }

    .summary-row:last-child {
        border-bottom: none;
    }

    .summary-label {
        color: #101828;
    }

    .summary-value {
        font-weight: 650;
        font-size: 13px;
        color: #101828;
    }

    /* ---------- Sidebar ---------- */
    .sidebar-note {
        color: #667085;
        font-size: 13px;
        line-height: 1.5;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        padding-top: 22px;
    }

    /* ---------- Remove unnecessary Streamlit decoration ---------- */
    [data-testid="stDecoration"] {
        display: none;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        min-height: 48px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "loan_approval_dataset.csv"
LOGISTIC_MODEL_FILE = "models/logistic_model.pkl"
TREE_MODEL_FILE = "models/decision_tree_model.pkl"

EXPECTED_FEATURES = [
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "applicantincome",
    "coapplicantincome",
    "loanamount",
    "loan_amount_term",
    "credit_history",
    "property_area",
]

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Predict loan approval using Logistic Regression and Decision Tree</div>',
    unsafe_allow_html=True,
)

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():
    if not os.path.exists(LOGISTIC_MODEL_FILE):
        raise FileNotFoundError(
            f"Logistic Regression model not found: {LOGISTIC_MODEL_FILE}"
        )

    if not os.path.exists(TREE_MODEL_FILE):
        raise FileNotFoundError(
            f"Decision Tree model not found: {TREE_MODEL_FILE}"
        )

    logistic_model = joblib.load(LOGISTIC_MODEL_FILE)
    tree_model = joblib.load(TREE_MODEL_FILE)

    return logistic_model, tree_model


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    return df


try:
    logistic_model, tree_model = load_models()
    df = load_dataset()
except Exception as error:
    st.error("The application could not load the trained models or dataset.")
    st.exception(error)
    st.stop()

# ============================================================
# VERIFY DATASET
# ============================================================

missing_features = [
    feature for feature in EXPECTED_FEATURES if feature not in df.columns
]

if missing_features:
    st.error("The dataset is missing the following features:")
    st.write(missing_features)
    st.stop()

if "loan_status" not in df.columns:
    st.error("The dataset does not contain the required target column: loan_status")
    st.stop()

# ============================================================
# MODEL PERFORMANCE
# ============================================================

@st.cache_data
def calculate_model_metrics(data):
    X = data[EXPECTED_FEATURES]
    y = data["loan_status"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logistic_predictions = logistic_model.predict(X_test)
    tree_predictions = tree_model.predict(X_test)

    return {
        "logistic_accuracy": accuracy_score(y_test, logistic_predictions),
        "logistic_precision": precision_score(
            y_test, logistic_predictions, zero_division=0
        ),
        "logistic_recall": recall_score(
            y_test, logistic_predictions, zero_division=0
        ),
        "logistic_f1": f1_score(
            y_test, logistic_predictions, zero_division=0
        ),
        "tree_accuracy": accuracy_score(y_test, tree_predictions),
        "tree_precision": precision_score(
            y_test, tree_predictions, zero_division=0
        ),
        "tree_recall": recall_score(
            y_test, tree_predictions, zero_division=0
        ),
        "tree_f1": f1_score(
            y_test, tree_predictions, zero_division=0
        ),
    }


metrics = calculate_model_metrics(df)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📊 Model Information")

    st.markdown(
        '<div class="sidebar-note">'
        "This application uses two supervised classification algorithms "
        "trained on the loan approval dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Logistic Regression")

    st.metric(
        "Accuracy",
        f"{metrics['logistic_accuracy'] * 100:.2f}%",
    )

    st.metric(
        "F1 Score",
        f"{metrics['logistic_f1'] * 100:.2f}%",
    )

    st.divider()

    st.subheader("Decision Tree")

    st.metric(
        "Accuracy",
        f"{metrics['tree_accuracy'] * 100:.2f}%",
    )

    st.metric(
        "F1 Score",
        f"{metrics['tree_f1'] * 100:.2f}%",
    )

    st.divider()

    st.caption(
        "The trained models are loaded from the models folder. "
        "The application does not retrain them during prediction."
    )

# ============================================================
# APPLICANT DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">👤 Applicant Details</div>',
    unsafe_allow_html=True,
)

st.info(
    "Enter the applicant's information below and click "
    "'Predict Loan Approval' to get predictions from both models."
)

left_column, right_column = st.columns(2, gap="large")

# ============================================================
# PERSONAL INFORMATION
# ============================================================

with left_column:
    st.markdown(
        '<div class="input-card-title">Personal Information</div>',
        unsafe_allow_html=True,
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        index=0,
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"],
        index=0,
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"],
        index=0,
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"],
        index=0,
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"],
        index=0,
    )

# ============================================================
# LOAN INFORMATION
# ============================================================

with right_column:
    st.markdown(
        '<div class="input-card-title">Loan Information</div>',
        unsafe_allow_html=True,
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=100,
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0,
        value=2000,
        step=100,
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=150,
        step=10,
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=12,
        value=360,
        step=12,
    )

    credit_history = st.selectbox(
        "Credit History",
        ["Yes", "No"],
        index=0,
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"],
        index=0,
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

predict_button = st.button(
    "🔮 Predict Loan Approval",
    type="primary",
    use_container_width=True,
)

# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Convert UI values to the same numeric encoding
    # used by the cleaned dataset.
    # --------------------------------------------------------

    user_data = pd.DataFrame(
        {
            "gender": [1 if gender == "Male" else 0],
            "married": [1 if married == "Yes" else 0],
            "dependents": [
                3 if dependents == "3+" else int(dependents)
            ],
            "education": [1 if education == "Graduate" else 0],
            "self_employed": [1 if self_employed == "Yes" else 0],
            "applicantincome": [applicant_income],
            "coapplicantincome": [coapplicant_income],
            "loanamount": [loan_amount],
            "loan_amount_term": [loan_term],
            "credit_history": [1 if credit_history == "Yes" else 0],
            "property_area": [
                {
                    "Rural": 0,
                    "Semiurban": 1,
                    "Urban": 2,
                }[property_area]
            ],
        }
    )

    user_data = user_data[EXPECTED_FEATURES]

    try:
        user_data = user_data.astype(float)

        logistic_prediction = int(logistic_model.predict(user_data)[0])
        tree_prediction = int(tree_model.predict(user_data)[0])

        logistic_probability = float(
            logistic_model.predict_proba(user_data)[0][1]
        )

        tree_probability = float(
            tree_model.predict_proba(user_data)[0][1]
        )

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
        st.stop()

    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Prediction Results</div>',
        unsafe_allow_html=True,
    )

    result_col1, result_col2 = st.columns(2, gap="large")

    # --------------------------------------------------------
    # Logistic Regression result
    # --------------------------------------------------------

    logistic_class = (
        "approved" if logistic_prediction == 1 else "rejected"
    )

    logistic_status_class = (
        "approved-text" if logistic_prediction == 1 else "rejected-text"
    )

    logistic_status = (
        "✓ LOAN APPROVED"
        if logistic_prediction == 1
        else "✗ LOAN REJECTED"
    )

    with result_col1:
        st.markdown(
            f"""
            <div class="prediction-card {logistic_class}">
                <div class="model-name">Logistic Regression</div>
                <div class="prediction-status {logistic_status_class}">
                    {logistic_status}
                </div>
                <div class="probability-label">
                    Approval Probability
                </div>
                <div class="probability-value">
                    {logistic_probability * 100:.2f}%
                </div>
                <div class="probability-track">
                    <div class="probability-fill"
                         style="width:{logistic_probability * 100:.2f}%;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Decision Tree result
    # --------------------------------------------------------

    tree_class = "approved" if tree_prediction == 1 else "rejected"

    tree_status_class = (
        "approved-text" if tree_prediction == 1 else "rejected-text"
    )

    tree_status = (
        "✓ LOAN APPROVED"
        if tree_prediction == 1
        else "✗ LOAN REJECTED"
    )

    with result_col2:
        st.markdown(
            f"""
            <div class="prediction-card {tree_class}">
                <div class="model-name">Decision Tree</div>
                <div class="prediction-status {tree_status_class}">
                    {tree_status}
                </div>
                <div class="probability-label">
                    Approval Probability
                </div>
                <div class="probability-value">
                    {tree_probability * 100:.2f}%
                </div>
                <div class="probability-track">
                    <div class="probability-fill"
                         style="width:{tree_probability * 100:.2f}%;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    if logistic_prediction == tree_prediction:

        if logistic_prediction == 1:
            comparison_title = " Models Agreement"
            comparison_result = "Both models predict LOAN APPROVED."
        else:
            comparison_title = "Models Agreement"
            comparison_result = "Both models predict LOAN REJECTED."

        comparison_note = (
            "Both algorithms produced the same classification for this applicant."
        )

    else:
        comparison_title = "⚠ Models Disagree"
        comparison_result = (
            "The two models produced different predictions."
        )
        comparison_note = (
            "Review the individual model predictions and probabilities "
            "rather than treating either model as automatically correct."
        )

    st.markdown(
        f"""
        <div class="comparison-card">
            <div class="comparison-title">{comparison_title}</div>
            <div class="comparison-result">{comparison_result}</div>
            <div class="comparison-note">{comparison_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # APPLICANT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">📝 Applicant Summary</div>',
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2 = st.columns(2, gap="large")

    with summary_col1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-row">
                    <span class="summary-label">Gender</span><br>
                    <span class="summary-value">{gender}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Married</span><br>
                    <span class="summary-value">{married}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Dependents</span><br>
                    <span class="summary-value">{dependents}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Education</span><br>
                    <span class="summary-value">{education}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Self Employed</span><br>
                    <span class="summary-value">{self_employed}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_col2:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-row">
                    <span class="summary-label">Applicant Income</span><br>
                    <span class="summary-value">{applicant_income}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Co-applicant Income</span><br>
                    <span class="summary-value">{coapplicant_income}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Loan Amount</span><br>
                    <span class="summary-value">{loan_amount}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Loan Term</span><br>
                    <span class="summary-value">{loan_term} months</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Credit History</span><br>
                    <span class="summary-value">{credit_history}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Property Area</span><br>
                    <span class="summary-value">{property_area}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Machine Learning Project · Loan Approval Prediction
        using Logistic Regression and Decision Tree
    </div>
    """,
    unsafe_allow_html=True,
)
