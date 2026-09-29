import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

def encode_cyclical_month(df):
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}


    month_nums = df['month'].map(month_map) if 'month' in df.columns else df.iloc[:, 0].map(month_map)

    sin_month = np.sin(2 * np.pi * month_nums / 12)
    cos_month = np.cos(2 * np.pi * month_nums / 12)

    return np.c_[sin_month, cos_month]
# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Bank Customer Conversion Intelligence",
    page_icon="🏦",
    layout="wide"
)
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# =========================================================
# LOAD ASSETS
# =========================================================
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("model/preprocessor.pkl")
    model = xgb.XGBClassifier()
    model.load_model("model/xgb_model.json")
    return preprocessor, model


preprocessor, model = load_assets()
test_results = pd.read_csv("test_results.csv")
# =========================================================
# LOAD DATA
# =========================================================


import xgboost as xgb


model = xgb.XGBClassifier()  # استخدم XGBRegressor لو موديلك regression
model.load_model("xgb_model.json")
preprocessor = joblib.load("preprocessor.pkl")

test_results = pd.read_csv("test_results.csv")

THRESHOLD = 0.22

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Bank Conversion Intelligence")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Model Performance", "Customer Targeting", "Live Prediction"],
)

# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.title("🏦 Bank Customer Conversion Intelligence")

    st.markdown(
        """
        ### Predicting and prioritizing customers for term-deposit campaigns

        This system analyzes historical customer data and predicts
        which customers are more likely to subscribe to a term deposit.
        """
    )

    st.divider()

    total_customers = len(test_results)

    actual_yes = test_results["actual"].sum()

    conversion_rate = actual_yes / total_customers * 100

    avg_balance = test_results["balance"].mean()

    previous_contact_rate = (
        (test_results["previous"] > 0).mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Conversion Rate",
        f"{conversion_rate:.2f}%"
    )

    col3.metric(
        "Previous Contact",
        f"{previous_contact_rate:.1f}%"
    )

    col4.metric(
        "Average Balance",
        f"{avg_balance:,.0f}"
    )

    st.divider()

    st.subheader("Conversion by Job")

    job_conversion = (
        test_results
        .groupby("job")["actual"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(job_conversion)

    st.subheader("Conversion by Age Group")

    test_results["age_group"] = pd.cut(
        test_results["age"],
        bins=[0, 30, 40, 50, 60, np.inf],
        labels=[
            "<30",
            "30–39",
            "40–49",
            "50–59",
            "60+"
        ]
    )

    age_conversion = (
        test_results
        .groupby("age_group", observed=True)["actual"]
        .mean()
        .mul(100)
    )

    st.bar_chart(age_conversion)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    y_true = test_results["actual"]

    y_prob = test_results["probability"]

    y_pred = test_results["prediction"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}"
    )

    col2.metric(
        "PR-AUC",
        f"{pr_auc:.3f}"
    )

    col3.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Precision",
        f"{precision:.3f}"
    )

    col2.metric(
        "Recall",
        f"{recall:.3f}"
    )

    col3.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Prediction Probability Distribution")

    st.bar_chart(
        test_results["probability"]
        .round(1)
        .value_counts()
        .sort_index()
    )

# =========================================================
# CUSTOMER TARGETING
# =========================================================

elif page == "Customer Targeting":

    st.title("🎯 Customer Targeting")

    st.markdown(
        """
        Customers are ranked according to their predicted probability
        of subscribing to a term deposit.
        """
    )

    ranked = test_results.sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)

    total = len(ranked)

    top10 = ranked.iloc[:int(total * 0.10)]

    top20 = ranked.iloc[:int(total * 0.20)]

    baseline = ranked["actual"].mean()

    top10_conversion = top10["actual"].mean()

    top20_conversion = top20["actual"].mean()

    top10_lift = top10_conversion / baseline

    top20_lift = top20_conversion / baseline

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Baseline Conversion",
        f"{baseline * 100:.2f}%"
    )

    col2.metric(
        "Top 10% Conversion",
        f"{top10_conversion * 100:.2f}%"
    )

    col3.metric(
        "Top 10% Lift",
        f"{top10_lift:.2f}x"
    )

    col4.metric(
        "Top 20% Conversion",
        f"{top20_conversion * 100:.2f}%"
    )

    st.divider()

    st.subheader("Customer Ranking")

    display_columns = [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "loan",
        "contact",
        "previous",
        "poutcome",
        "probability",
        "prediction"
    ]

    available_columns = [
        col for col in display_columns
        if col in ranked.columns
    ]

    display_df = ranked[available_columns].copy()

    display_df["probability"] = (
        display_df["probability"] * 100
    ).round(2)

    display_df = display_df.rename(
        columns={
            "probability": "Conversion Probability (%)"
        }
    )

    st.dataframe(
        display_df.head(100),
        use_container_width=True
    )

    st.caption(
        "Customers are ranked by predicted conversion probability."
    )

# =========================================================
# LIVE PREDICTION
elif page == "Live Prediction":
    st.title("🔮 Single Customer Live Prediction")
    st.write(
        "Enter customer details below to predict term deposit subscription probability."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox(
                "Job",
                [
                    "admin.",
                    "blue-collar",
                    "technician",
                    "services",
                    "management",
                    "retired",
                    "entrepreneur",
                    "self-employed",
                    "unemployed",
                    "housemaid",
                    "student",
                ],
            )
            marital = st.selectbox(
                "Marital Status", ["single", "married", "divorced"]
            )
            education = st.selectbox(
                "Education", ["primary", "secondary", "tertiary", "unknown"]
            )
            default = st.selectbox("Credit in Default?", ["no", "yes"])

        with col2:
            balance = st.number_input("Account Balance (€)", value=1500)
            housing = st.selectbox("Housing Loan", ["yes", "no"])
            loan = st.selectbox("Personal Loan", ["yes", "no"])
            contact = st.selectbox(
                "Contact Type", ["cellular", "telephone", "unknown"]
            )
            day = st.slider("Last Contact Day of Month", 1, 31, 15)

        with col3:
            month = st.selectbox(
                "Last Contact Month",
                [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ],
            )
            previous = st.number_input(
                "Previous Contacts", min_value=0, value=1
            )
            pdays = st.number_input(
                "Days Since Last Campaign (pdays)", value=-1
            )
            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                ["unknown", "failure", "other", "success"],
            )

        submit = st.form_submit_button("Predict Conversion")

    if submit:
        # إنشاء DataFrame بجميع الأعمدة المطلوبة بالكامل
        input_data = pd.DataFrame(
            [{
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "day": day,
                "month": month,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
            }]
        )

        try:
            # تحويل البيانات واستخراج التوقع
            X_input = preprocessor.transform(input_data)
            prob = model.predict_proba(X_input)[0][1]

            st.divider()
            st.subheader("Prediction Result:")
            st.metric("Conversion Probability", f"{prob * 100:.2f}%")

            if prob >= 0.22:
                st.success(
                    "🎯 High Potential Customer! Recommended for Campaign."
                )
            else:
                st.warning(
                    "⚠️ Low Probability. Lower priority for direct contact."
                )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
