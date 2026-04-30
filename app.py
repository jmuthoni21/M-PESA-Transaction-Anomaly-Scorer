from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

from lime.lime_tabular import LimeTabularExplainer

# ─────────────────────────────────────────────
# 1. Load model + schema
# ─────────────────────────────────────────────
model = joblib.load("ml/fraud_model.pkl")
feature_cols = joblib.load("ml/feature_columns.pkl")

# ─────────────────────────────────────────────
# 2. App init
# ─────────────────────────────────────────────
app = FastAPI(title="M-PESA Fraud Detection API", version="3.0")

# ─────────────────────────────────────────────
# 3. Request schema
# ─────────────────────────────────────────────
class Transaction(BaseModel):
    amount_kes: float
    hour: int
    day_of_week: int
    transaction_type: str
    user_avg_amount: float = 2000
    time_diff: float = 0


# ─────────────────────────────────────────────
# 4. Feature builder
# ─────────────────────────────────────────────
def build_features(data: dict) -> pd.DataFrame:

    amount_kes = data["amount_kes"]
    hour = data["hour"]
    day_of_week = data["day_of_week"]
    transaction_type = data["transaction_type"]

    user_avg_amount = max(data.get("user_avg_amount", 1), 1)
    time_diff = max(data.get("time_diff", 0), 0)

    is_night = int(hour >= 22 or hour <= 5)
    is_weekend = int(day_of_week >= 5)

    log_amount = np.log1p(amount_kes)
    is_large_tx = int(amount_kes > 15000)

    amount_deviation = min(amount_kes / user_avg_amount, 10)
    time_diff = min(time_diff, 100000)

    features = {
        "amount_kes": float(amount_kes),
        "log_amount": float(log_amount),
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "is_night": int(is_night),
        "is_weekend": int(is_weekend),
        "is_large_tx": int(is_large_tx),
        "user_avg_amount": float(user_avg_amount),
        "amount_deviation": float(amount_deviation),
        "time_diff": float(time_diff)
    }

    valid_types = [
        c.replace("transaction_type_", "")
        for c in feature_cols
        if c.startswith("transaction_type_")
    ]

    if transaction_type not in valid_types:
        raise ValueError(f"Invalid transaction_type. Must be one of {valid_types}")

    for col in feature_cols:
        if col.startswith("transaction_type_"):
            features[col] = int(col == f"transaction_type_{transaction_type}")

    X = pd.DataFrame([features])

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    return X[feature_cols]


# ─────────────────────────────────────────────
# 5. Load training data for LIME
# ─────────────────────────────────────────────
train_df = pd.read_csv("data/transactions.csv")

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
train_df["hour"] = train_df["timestamp"].dt.hour
train_df["day_of_week"] = train_df["timestamp"].dt.dayofweek

train_df["time_diff"] = train_df["time_diff"].fillna(0)
train_df["user_avg_amount"] = train_df["user_avg_amount"].replace(0, 1)

train_df["log_amount"] = np.log1p(train_df["amount_kes"])
train_df["is_night"] = ((train_df["hour"] >= 22) | (train_df["hour"] <= 5)).astype(int)
train_df["is_weekend"] = (train_df["day_of_week"] >= 5).astype(int)
train_df["is_large_tx"] = (train_df["amount_kes"] > 15000).astype(int)

train_df["amount_deviation"] = (
    train_df["amount_kes"] / train_df["user_avg_amount"]
).clip(0, 10)

train_df = pd.get_dummies(train_df, columns=["transaction_type"], drop_first=True)

for col in feature_cols:
    if col not in train_df.columns:
        train_df[col] = 0

X_train = train_df[feature_cols]


# ─────────────────────────────────────────────
# 6. LIME explainer
# ─────────────────────────────────────────────
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_cols,
    class_names=["legit", "fraud"],
    mode="classification"
)


def get_lime_explanation(X_row):
    try:
        exp = lime_explainer.explain_instance(
            X_row.values[0],
            model.predict_proba
        )

        return [
            {
                "feature": str(f),
                "impact": float(v)
            }
            for f, v in exp.as_list()
        ]

    except Exception:
        return [{"feature": "explanation_error", "impact": 0.0}]


# ─────────────────────────────────────────────
# 7. Human explanation engine
# ─────────────────────────────────────────────
def build_fraud_story(tx: dict, lime_explanations: list):

    stories = []

    amount = tx["amount_kes"]
    avg = tx.get("user_avg_amount", amount)
    hour = tx["hour"]
    time_diff = tx.get("time_diff", 0)

    # time patterns
    if hour >= 22 or hour <= 5:
        stories.append("Transaction occurred during unusual night-time hours.")

    # spending spike
    if amount > avg * 2:
        multiplier = round(amount / avg, 1)
        stories.append(f"Transaction is {multiplier}× higher than user's normal spending behavior.")

    # rapid activity
    if time_diff < 120:
        stories.append("Multiple transactions detected in a very short time window.")

    # weekend behavior
    if tx["day_of_week"] >= 5:
        stories.append("Activity detected during weekend pattern shift.")

    # strongest model signals
    top = sorted(lime_explanations, key=lambda x: abs(x["impact"]), reverse=True)[:2]

    for t in top:
        f = t["feature"]

        if "night" in f:
            stories.append("Model detected unusual timing behavior.")
        elif "amount" in f:
            stories.append("Spending pattern deviates significantly from normal behavior.")
        elif "time_diff" in f:
            stories.append("Transaction frequency is unusually high.")

    # remove duplicates
    return list(dict.fromkeys(stories))


# ─────────────────────────────────────────────
# 8. Routes
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "M-PESA Fraud Detection API is running"}


@app.post("/score")
def score(tx: Transaction):

    X = build_features(tx.dict())
    prob = float(model.predict_proba(X)[0][1])

    return {
        "fraud_probability": prob,
        "is_flagged": bool(prob > 0.4)
    }


@app.post("/score_explain")
def score_explain(tx: Transaction):

    X = build_features(tx.dict())

    prob = float(model.predict_proba(X)[0][1])

    lime_raw = get_lime_explanation(X)

    story = build_fraud_story(tx.dict(), lime_raw)

    if prob > 0.7:
        risk_level = "HIGH_RISK"
    elif prob > 0.4:
        risk_level = "MEDIUM_RISK"
    else:
        risk_level = "LOW_RISK"

    return {
        "fraud_probability": prob,
        "risk_level": risk_level,
        "is_flagged": bool(prob > 0.4),

        #THIS is what your UI should use
        "explanation_story": story,

        # optional debug (safe now)
        "lime_raw": lime_raw
    }