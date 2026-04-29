import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load data ──────────────────────────────
df = pd.read_csv('data/transactions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Dataset shape:", df.shape)
print("Fraud rate:", df['is_fraud'].mean().round(4))

# ── 2. Feature Engineering (minimal — dataset already rich) ──
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Handle missing values (important)
df['time_diff'] = df['time_diff'].fillna(df['time_diff'].median())

# Log transform for stability
df['log_amount'] = np.log1p(df['amount_kes'])

# Optional: cap extreme deviations (prevents crazy values dominating)
df['amount_deviation'] = df['amount_deviation'].clip(0, 10)

# ── 3. Encode categorical features ────────────
df = pd.get_dummies(df, columns=['transaction_type'], drop_first=True)

# ── 4. Feature selection ──────────────────────
feature_cols = [
    'amount_kes',
    'log_amount',
    'hour',
    'day_of_week',
    'is_night',
    'is_weekend',
    'is_large_tx',
    'user_avg_amount',
    'amount_deviation',
    'time_diff'
] + [col for col in df.columns if col.startswith('transaction_type_')]

X = df[feature_cols]
y = df['is_fraud']

# ── 5. Train/test split ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"\nTraining set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# ── 6. SMOTE (only on training set) ───────────
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {X_train_bal.shape[0]} rows")
print(f"Fraud rate after SMOTE: {y_train_bal.mean():.2%}")

# ── 7. Train model ───────────────────────────
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train_bal, y_train_bal)

# ── 8. Evaluate ──────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ── 9. Feature importance ─────────────────────
importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Feature Importance — M-PESA Fraud Model')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)

print("\nSaved feature_importance.png")

# ── 10. Save artifacts ───────────────────────
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(feature_cols, 'feature_columns.pkl')

print("Model and feature columns saved")
