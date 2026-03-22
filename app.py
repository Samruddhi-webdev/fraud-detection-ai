import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ---------- CSS Styling ----------
st.markdown("""
<style>

body {
    background-color: #111827;
}

.stApp {
    background: linear-gradient(135deg,#111827,#1f2937);
    color:white;
}

label {
    color: white !important;
    font-weight: 600;
}

h1, h2, h3 {
    color:white !important;
}

[data-testid="stMetricValue"] {
    color:white !important;
    font-size:40px;
}

.stButton>button {
    background-color:#ef4444;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
}

.stButton>button:hover {
    background-color:#dc2626;
}
/* SUCCESS BOX (System Status) */
.stSuccess {
    background-color:#065f46 !important;
    color:#ffffff !important;
    font-weight:600;
}

/* WARNING BOX (Risk Factors) */
.stWarning {
    background-color:#92400e !important;
    color:#ffffff !important;
    font-weight:600;
}

/* ERROR BOX */
.stError {
    background-color:#7f1d1d !important;
    color:#ffffff !important;
    font-weight:600;
}

/* Rounded alert boxes */
.stAlert {
    border-radius:10px;
    padding:12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
model = joblib.load("fraud_model.pkl")

# ---------- Title ----------
st.title("🛡️ AI Fraud Detection System")
st.markdown("### 🔍 Real-Time Transaction Risk Analyzer")

st.write(
"This system analyzes transaction behavior using machine learning "
"to detect potential fraudulent activity."
)

# ---------- Transaction Inputs ----------
st.subheader("📊 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("💰 Transaction Amount", min_value=0)
    foreign_transaction = st.selectbox("🌍 Foreign Transaction", [0,1])
    device_trust_score = st.number_input(
        "Device Trust Score",
        min_value=0.0,
        max_value=1.0,
        step=0.1
    )
    cardholder_age = st.number_input(
        "Cardholder Age",
        min_value=18,
        max_value=100,
        step=1
    )

with col2:
    transaction_hour = st.number_input(
        "🕒 Transaction Hour (0-23)",
        min_value=0,
        max_value=23,
        step=1
    )
    location_mismatch = st.selectbox("📍 Location Mismatch", [0,1])
    velocity_last_24h = st.number_input(
        "Transactions in last 24h",
        min_value=0,
        step=1
    )

# ---------- Prediction ----------
if st.button("Check Fraud"):

    input_data = pd.DataFrame([[amount,
                                transaction_hour,
                                foreign_transaction,
                                location_mismatch,
                                device_trust_score,
                                velocity_last_24h,
                                cardholder_age]],
                               columns=[
                                   "amount",
                                   "transaction_hour",
                                   "foreign_transaction",
                                   "location_mismatch",
                                   "device_trust_score",
                                   "velocity_last_24h",
                                   "cardholder_age"
                               ])

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    risk_score = prob * 100

    # ---------- Top Dashboard Row ----------
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Fraud Risk Score", f"{risk_score:.2f}%")

    with colB:
        if risk_score < 30:
            st.success("🟢 Risk Level: Low")
        elif risk_score < 70:
            st.warning("🟡 Risk Level: Medium")
        else:
            st.error("🔴 Risk Level: High")

    with colC:
        if prediction[0] == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Transaction Safe")

    # ---------- Graph + Summary ----------
    left, right = st.columns(2)

    with left:
        st.subheader("📊 Risk Visualization")

        fig, ax = plt.subplots()

        labels = ['Safe Probability', 'Fraud Probability']
        values = [100-risk_score, risk_score]

        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title("Transaction Risk Distribution")

        st.pyplot(fig)

    with right:
        st.subheader("📄 Transaction Summary")

        st.write({
            "Amount": amount,
            "Transaction Hour": transaction_hour,
            "Foreign Transaction": foreign_transaction,
            "Location Mismatch": location_mismatch,
            "Device Trust Score": device_trust_score,
            "Transactions Last 24h": velocity_last_24h,
            "Cardholder Age": cardholder_age
        })

# ---------- Risk Factors + Status ----------
left2, right2 = st.columns(2)

with left2:
    st.subheader("🔎 Risk Factors Analysis")

    if foreign_transaction == 1:
        st.warning("Foreign transaction detected")

    if location_mismatch == 1:
        st.warning("Location mismatch detected")

    if transaction_hour <= 4:
        st.warning("Late-night transaction pattern")

    if velocity_last_24h > 5:
        st.warning("High number of transactions in last 24 hours")

    if device_trust_score < 0.4:
        st.warning("Low device trust score")

with right2:
    st.subheader("🛡️ System Status")
    st.success("Fraud detection engine active")

# ---------- Footer ----------
st.markdown("---")
st.caption(
"AI Fraud Detection System | Machine Learning Powered | Real-Time Risk Analysis"
)
    

    