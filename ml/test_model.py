import numpy as np
import joblib
import pandas as pd

# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
model = joblib.load('models/fraud_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')


def score_transaction(
    amount_kes: float,
    hour: int,
    day_of_week: int,
    transaction_type: str,
    user_avg_amount: float,
    time_diff: float
) -> float:
    """
    Scores a transaction using the trained fraud model.

    Parameters:
    - amount_kes (float): Transaction amount
    - hour (int): Hour of day (0–23)
    - day_of_week (int): Day (0=Mon, 6=Sun)
    - transaction_type (str): Transaction category
    - user_avg_amount (float): User's typical transaction amount
    - time_diff (float): Seconds since last transaction

    Returns:
    - float: Fraud probability (0 to 1)
    """

    # ── Safety guards ─────────────────────────
    user_avg_amount = max(user_avg_amount, 1)  # avoid division by zero
    time_diff = max(time_diff, 0)

    # ── Feature engineering (MATCH TRAINING) ──
    is_night = int(hour >= 22 or hour <= 5)
    is_weekend = int(day_of_week >= 5)
    log_amount = np.log1p(amount_kes)
    is_large_tx = int(amount_kes > 15000)

    amount_deviation = amount_kes / user_avg_amount
    amount_deviation = min(amount_deviation, 10)

    # ── Base feature dict ─────────────────────
    data = {
        'amount_kes': amount_kes,
        'log_amount': log_amount,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_night': is_night,
        'is_weekend': is_weekend,
        'is_large_tx': is_large_tx,
        'user_avg_amount': user_avg_amount,
        'amount_deviation': amount_deviation,
        'time_diff': time_diff
    }

    # ── One-hot encoding (CRITICAL) ───────────
    for col in feature_cols:
        if col.startswith('transaction_type_'):
            data[col] = int(col == f'transaction_type_{transaction_type}')

    # ── Build dataframe ───────────────────────
    X = pd.DataFrame([data])

    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    # Enforce correct order
    X = X[feature_cols]

    # ── Prediction ────────────────────────────
    fraud_prob = model.predict_proba(X)[0][1]

    return float(fraud_prob)


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Normal transaction
    score_normal = score_transaction(
        amount_kes=1500,
        hour=14,
        day_of_week=2,
        transaction_type='paybill',
        user_avg_amount=2000,
        time_diff=3600
    )

    print(f"Normal transaction fraud score: {score_normal:.3f}")

    # Suspicious transaction
    score_suspicious = score_transaction(
        amount_kes=65000,
        hour=2,
        day_of_week=6,
        transaction_type='send_money',
        user_avg_amount=2000,
        time_diff=20
    )

    print(f"Suspicious transaction fraud score: {score_suspicious:.3f}")

    # Threshold decision
    threshold = 0.4
    print(f"\nIs suspicious transaction flagged? {score_suspicious > threshold}")