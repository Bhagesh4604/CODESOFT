# Cyber-Sentinel | Credit Card Fraud Protection

A mission-critical security dashboard that monitors and identifies fraudulent credit card transactions using weighted random forest heuristics.

## How it Works
Due to the extreme rarity of fraudulent transactions, this project utilizes **RandomOverSampler** (similar to SMOTE) to ensure the model detects malicious signatures effectively. The system is trained on a high-fidelity synthetic dataset mimicking the `mlg-ulb/creditcardfraud` PCA structure (V1-V28 vectors).

## Key Features
- **Tactical UI/UX:** Dark-mode security terminal with Matrix green/Red status alerts.
- **Deep-Scan Animation:** Simulated packet inspection during transaction processing.
- **Risk Assessment:** Real-time probability scoring and verdict console.

## Technologies Used
- **Python**: Scalable machine learning and data simulation.
- **Streamlit**: Advanced CSS-customized tactical dashboard.
- **Scikit-Learn**: Robust classification logic.
- **Imbalanced-Learn**: Handling extreme class skewness.

## How to Run
1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib imbalanced-learn
   ```
2. Initialize security assets:
   ```bash
   python fraud_model.py
   ```
3. Boot the Cyber-Sentinel core:
   ```bash
   streamlit run app.py
   ```

Developed by **Bhagesh Biradar**
