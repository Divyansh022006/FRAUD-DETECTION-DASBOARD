import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI-Powered Financial Fraud Detection Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("💳 AI-Powered Financial Fraud Detection Dashboard")
st.caption("Developed by Divyansh Agarwal")

st.markdown("""
### About Project

This system detects fraudulent credit card transactions using Machine Learning.

The model was trained using Logistic Regression and deployed using Streamlit.

Upload a transaction dataset and view fraud analytics instantly.
""")

# ---------------- LOAD MODEL ----------------

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.header("📊 Model Performance")

st.sidebar.success("Accuracy: 99.92%")
st.sidebar.success("Precision: 83%")
st.sidebar.success("Recall: 64%")
st.sidebar.success("F1 Score: 72%")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📁 Upload CSV File",
    type=["csv"]
)

# ---------------- PROCESS FILE ----------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data")
    st.dataframe(df.head())

    # SCALE DATA

    scaled_data = scaler.transform(df)

    # PREDICTIONS

    predictions = model.predict(scaled_data)

    probabilities = model.predict_proba(scaled_data)

    # ADD RESULTS TO DATAFRAME

    df["Prediction"] = predictions

    df["Fraud Probability (%)"] = (
        probabilities[:, 1] * 100
    ).round(2)

    # COUNTS

    total_count = len(df)

    fraud_count = (df["Prediction"] == 1).sum()

    genuine_count = (df["Prediction"] == 0).sum()

    # ALERT

    if fraud_count > 0:
        st.error(
            f"⚠️ Warning: {fraud_count} Fraudulent Transaction(s) Detected"
        )
    else:
        st.success(
            "✅ No Fraudulent Transactions Found"
        )

    # METRICS

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Transactions",
        total_count
    )

    col2.metric(
        "Fraud Transactions",
        fraud_count
    )

    col3.metric(
        "Legitimate Transactions",
        genuine_count
    )

    # RESULTS TABLE

    st.subheader("🤖 Prediction Results")

    st.dataframe(df)

    # FRAUD ONLY TABLE

    st.subheader("🚨 Detected Fraud Transactions")

    fraud_df = df[df["Prediction"] == 1]

    if len(fraud_df) > 0:
        st.dataframe(fraud_df)
    else:
        st.success(
            "No Fraud Transactions Found"
        )

    # PIE CHART

    st.subheader("🥧 Fraud Distribution")

    fig, ax = plt.subplots()

    ax.pie(
        [fraud_count, genuine_count],
        labels=["Fraud", "Legitimate"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    # BAR CHART

    st.subheader("📈 Transaction Summary")

    chart_data = pd.DataFrame({
        "Category": ["Fraud", "Legitimate"],
        "Count": [fraud_count, genuine_count]
    })

    st.bar_chart(
        chart_data.set_index("Category")
    )

    # DOWNLOAD REPORT

    csv = df.to_csv(index=False)

    st.download_button(
    label="📥 Download Fraud Analysis Report",
    data=csv,
    file_name="fraud_report.csv",
    mime="text/csv"
)
    st.markdown("---")

st.markdown(
    "🚀 Built using Python, Streamlit, Scikit-Learn, Pandas and Machine Learning"
)