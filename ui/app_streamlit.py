import streamlit as st
import requests

# ─────────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────────
API_URL = "https://m-pesa-transaction-anomaly-scorer.onrender.com/score_explain"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Lab — M-PESA Guard",
    page_icon="💳",
    layout="centered"
)

# ─────────────────────────────────────────────
# THEME (soft geek girl aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>

body {
    background-color: #faf9ff;
}

.big-title {
    font-size: 34px;
    font-weight: 800;
    color: #4b3f72;
    letter-spacing: 0.5px;
}

.subtext {
    color: #6b6b7a;
    font-size: 14px;
    margin-bottom: 20px;
}

.card {
    padding: 22px;
    border-radius: 18px;
    background: white;
    box-shadow: 0 6px 18px rgba(80, 70, 120, 0.08);
    border: 1px solid #eee;
}

.badge-high {
    background: #ff4b6e;
    color: white;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}

.badge-medium {
    background: #f5a623;
    color: white;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}

.badge-low {
    background: #2ecc71;
    color: white;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}

.section-title {
    font-size: 16px;
    font-weight: 700;
    margin-top: 15px;
    color: #3d3b4f;
}

.story {
    background: #f6f5ff;
    padding: 10px 12px;
    border-radius: 12px;
    margin: 6px 0;
    font-size: 14px;
    color: #3d3b4f;
}

.slider-label {
    font-size: 13px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="big-title">💳 Fraud Lab: M-PESA Guard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">A playful-but-serious ML system that explains suspicious transactions.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
with st.form("fraud_form"):

    amount_kes = st.number_input("💰 Transaction Amount (KES)", min_value=1.0, value=1500.0)
    hour = st.slider("🕒 Hour of Day", 0, 23, 14)
    day_of_week = st.slider("📅 Day (0=Mon, 6=Sun)", 0, 6, 2)

    transaction_type = st.selectbox(
        "🔁 Transaction Type",
        ["send_money", "paybill", "buy_goods", "withdraw", "deposit"]
    )

    user_avg_amount = st.number_input("📊 User Average Spend", value=2000.0)
    time_diff = st.number_input("⏱ Time Since Last Transaction (sec)", value=3600.0)

    submitted = st.form_submit_button("🔍 Analyze Transaction")

# ─────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────
def call_api(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
if submitted:

    payload = {
        "amount_kes": amount_kes,
        "hour": hour,
        "day_of_week": day_of_week,
        "transaction_type": transaction_type,
        "user_avg_amount": user_avg_amount,
        "time_diff": time_diff
    }

    try:
        result = call_api(payload)

        prob = result["fraud_probability"]
        risk = result["risk_level"]
        story = result["explanation_story"]

        

        # ── Risk Badge ─────────────────────────
        st.markdown("### 🧠 Fraud Risk Score")


        if risk == "HIGH_RISK":
            st.error(f"HIGH RISK — {prob:.2f}")

        elif risk == "MEDIUM_RISK":
            st.warning(f"MEDIUM RISK — {prob:.2f}")

        else:
            st.success(f"LOW RISK — {prob:.2f}")

        # ── Explanation Story ───────────────────
        st.markdown("### 🧾 Why this was flagged")

        if story:
            for s in story:
                st.markdown(f"<div class='story'>• {s}</div>", unsafe_allow_html=True)
        else:
            st.markdown("No unusual patterns detected.")

        # ── Debug Expand (optional nerd layer) ─
        with st.expander("🔬 Technical details (LIME raw output)"):
            st.json(result.get("lime_raw", []))

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("Could not connect to fraud API.")
        st.exception(e)