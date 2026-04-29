import numpy as np
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('fraud_model.pkl')
feature_cols = joblib.load('feature_columns.pkl')

def score_transaction(
    amount_kes,
    hour,
    day_of_week,
    transaction_type,
    user_avg_amount,
    time_diff
):
    """
    Scores a transaction using the trained fraud model.
    Returns fraud probability (0 to 1).
    """

    # --- Base features ---
    is_night = int(hour >= 22 or hour <= 5)
    is_weekend = int(day_of_week >= 5)
    log_amount = np.log1p(amount_kes)
    is_large_tx = int(amount_kes > 15000)

    # Behavioral feature
    amount_deviation = amount_kes / user_avg_amount if user_avg_amount > 0 else 0
    amount_deviation = min(amount_deviation, 10)

    # Build feature dict
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

    # --- Add one-hot transaction types ---
    for col in feature_cols:
        if col.startswith('transaction_type_'):
            data[col] = 1 if col == f'transaction_type_{transaction_type}' else 0

    # Create DataFrame with correct column order
    X = pd.DataFrame([data])

    # Ensure all columns exist (in case of missing ones)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]

    # Predict
    fraud_prob = model.predict_proba(X)[0][1]
    return fraud_prob


# ── Example usage ─────────────────────────────

# Normal transaction
score_normal = score_transaction(
    amount_kes=1500,
    hour=14,
    day_of_week=2,
    transaction_type='paybill',
    user_avg_amount=2000,
    time_diff=3600  # 1 hour since last transaction
)

print(f"Normal transaction fraud score: {score_normal:.3f}")

# Suspicious transaction
score_suspicious = score_transaction(
    amount_kes=65000,
    hour=2,
    day_of_week=6,
    transaction_type='send_money',
    user_avg_amount=2000,
    time_diff=20  # very rapid transaction
)

print(f"Suspicious transaction fraud score: {score_suspicious:.3f}")

# Decision threshold
threshold = 0.4
print(f"\nIs suspicious transaction flagged? {score_suspicious > threshold}")