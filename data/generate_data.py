import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_mpesa_data(n_transactions=10000, n_users=2000):

    # --- Split legit vs fraud ---
    n_legit = int(n_transactions * 0.97)
    n_fraud = n_transactions - n_legit

    # --- Legitimate transactions ---
    legit_hours_list = list(range(7, 21))  # 7AM → 8PM (14 values)

    # FIX: match probability length (14 values)
    legit_probs = np.array([
        0.03, 0.05, 0.07, 0.09, 0.10, 0.11, 0.11,
        0.10, 0.09, 0.08, 0.07, 0.05, 0.03, 0.02
    ])
    legit_probs = legit_probs / legit_probs.sum()

    legit_hours = np.random.choice(
        legit_hours_list,
        size=n_legit,
        p=legit_probs
    )

    legit_amounts = np.random.lognormal(mean=7.5, sigma=1.2, size=n_legit)
    legit_amounts = np.clip(legit_amounts, 10, 70000)

    # --- Fraudulent transactions ---
    fraud_hours = np.random.choice([0, 1, 2, 3, 22, 23], size=n_fraud)

    fraud_amounts = np.random.lognormal(mean=10.2, sigma=0.8, size=n_fraud)
    fraud_amounts = np.clip(fraud_amounts, 15000, 70000)

    # --- Combine ---
    hours = np.concatenate([legit_hours, fraud_hours])
    amounts = np.concatenate([legit_amounts, fraud_amounts])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # --- Timestamps ---
    base_date = datetime(2024, 1, 1)
    timestamps = [
        base_date + timedelta(days=random.randint(0, 365), hours=int(h))
        for h in hours
    ]

    df = pd.DataFrame({
        'transaction_id': [f'TXN{i:07d}' for i in range(n_transactions)],
        'timestamp': timestamps,
        'amount_kes': amounts.round(2),
        'hour_of_day': hours.astype(int),
        'is_fraud': labels.astype(int)
    })

    # --- Transaction types ---
    tx_types = ['send_money', 'paybill', 'buy_goods', 'withdraw', 'deposit']

    legit_weights = [0.25, 0.3, 0.25, 0.1, 0.1]
    fraud_weights = [0.6, 0.15, 0.1, 0.1, 0.05]

    df['transaction_type'] = ''
    df.loc[df['is_fraud'] == 0, 'transaction_type'] = np.random.choice(
        tx_types, size=n_legit, p=legit_weights
    )
    df.loc[df['is_fraud'] == 1, 'transaction_type'] = np.random.choice(
        tx_types, size=n_fraud, p=fraud_weights
    )

    # --- Time-based features ---
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day

    # --- Weekend behavior ---
    is_weekend = df['day_of_week'] >= 5
    mask = is_weekend & (df['is_fraud'] == 0)

    df.loc[mask, 'transaction_type'] = np.random.choice(
        ['send_money', 'buy_goods', 'withdraw'],
        size=mask.sum(),
        p=[0.5, 0.3, 0.2]
    )

    # --- Payday effect ---
    is_payday = df['day_of_month'] >= 25
    df.loc[is_payday & (df['is_fraud'] == 0), 'amount_kes'] *= np.random.uniform(1.5, 2.5)
    df['amount_kes'] = df['amount_kes'].clip(10, 70000)

    # --- Users ---
    user_ids = [f'USER{i:05d}' for i in range(n_users)]
    df['user_id'] = np.random.choice(user_ids, size=len(df))

    # --- User behavior ---
    user_avg = {user: np.random.uniform(200, 5000) for user in user_ids}
    df['user_avg_amount'] = df['user_id'].map(user_avg)

    df['amount_deviation'] = df['amount_kes'] / df['user_avg_amount']

    # --- Smarter fraud patterns ---
    fraud_idx = df['is_fraud'] == 1

    df.loc[fraud_idx, 'amount_kes'] *= np.random.choice(
        [0.5, 1, 3],
        size=fraud_idx.sum(),
        p=[0.2, 0.5, 0.3]
    )

    df.loc[fraud_idx, 'amount_deviation'] *= np.random.uniform(
        2, 6, size=fraud_idx.sum()
    )

    df['amount_kes'] = df['amount_kes'].clip(10, 70000)

    # --- Transaction bursts ---
    df = df.sort_values(['user_id', 'timestamp'])

    df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()

    df.loc[fraud_idx, 'time_diff'] = np.random.uniform(
        5, 60, size=fraud_idx.sum()
    )

    # --- Extra ML features ---
    df['is_night'] = df['hour_of_day'].isin([0, 1, 2, 3, 22, 23]).astype(int)
    df['is_large_tx'] = (df['amount_kes'] > 15000).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # --- Shuffle ---
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


if __name__ == '__main__':
    df = generate_mpesa_data(10000)

    df.to_csv('data/transactions.csv', index=False)

    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(df.head())