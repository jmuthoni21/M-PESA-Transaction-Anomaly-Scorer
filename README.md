# M-PESA-Transaction-Anomaly-Scorer
Build a fraud detection model on synthetic M-PESA transaction data. 


flowchart LR

    A[Incoming Transaction] --> B[API Endpoint]

    B --> C[Feature Engineering]
    C --> D[Behavioral Features]

    D --> E[Random Forest Model]
    E --> F[Fraud Probability]

    D --> G[SHAP Explainer]
    G --> H[Top Risk Factors]

    F --> I[Decision Layer]
    H --> I

    I --> J[Streamlit Dashboard]