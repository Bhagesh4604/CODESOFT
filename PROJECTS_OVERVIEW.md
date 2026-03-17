# CODESOFT Data Science Projects Overview

This document summarizes the four analytical projects completed during the Data Science Internship, highlighting the datasets, technical approaches, and core mechanics of each system.

---

## 1. Titanic Survival Predictor
**Goal:** Predict the survival probability of passengers on the Titanic based on historical data.

*   **Dataset:** `yasserh/titanic-dataset` (historical HMS Titanic passenger logs).
*   **Approach:** 
    *   **Preprocessing:** Used `StandardScaler` for numeric values and categorical encoding for titles and gender.
    *   **Model:** Random Forest Classifier (within a `Pipeline`).
*   **Sample Data:**
    *   `Age`: 25.0
    *   `Sex`: female
    *   `Pclass`: 1st Class
    *   `Fare`: £71.28
*   **How it Works:** The model identifies correlations between demographic features and survival. For example, it learned that women and children in higher classes had significantly higher survival rates, allowing it to predict destiny for new inputs.
*   **Aesthetic:** "Dark Ocean" (Deep navy and teal UI).

---

## 2. Movie Rating Predictor
**Goal:** Estimate the IMDb rating of Indian movies based on production attributes.

*   **Dataset:** IMDb India Movies dataset (historical movie ratings, directors, and actors).
*   **Approach:**
    *   **Algorithm:** Random Forest Regression.
    *   **Custom Logic:** Weighted Similarity Engine for recommendations.
*   **Sample Data:**
    *   `Year`: 2023
    *   `Director`: Rajkumar Hirani
    *   `Genre`: Drama
    *   `Votes`: 15,000
*   **How it Works:** It analyzes the historical performance of specific directors and actors within genres to forecast a rating. The "Similar Movies" feature calculates a score using a weighting system (Genre=3 pts, Director=4 pts, etc.) to offer tailored recommendations.
*   **Aesthetic:** "Magenta/Burgundy" (Premium cinematic dashboard).

---

## 3. Iris Flower Classification
**Goal:** Classify iris flowers into three species based on botanical measurements.

*   **Dataset:** `arshid/iris-flower-dataset` (Fisher's Iris data).
*   **Approach:**
    *   **Model:** Random Forest Classifier.
    *   **State Management:** Streamlit Session State for instant updates.
*   **Sample Data:**
    *   `Sepal Length`: 5.1 cm
    *   `Sepal Width`: 3.5 cm
    *   `Petal Length`: 1.4 cm
    *   `Petal Width`: 0.2 cm
*   **How it Works:** It classifies flowers into **Setosa**, **Versicolor**, or **Virginica**. Because the species are highly distinct in their petal/sepal ratios, the model achieves near 100% accuracy by finding simple linear and non-linear boundaries in the feature space.
*   **Aesthetic:** "Botanical Glassmorphic" (Clean, scientific frosted-glass UI).

---

## 4. Sales Prediction Engine
**Goal:** Forecast product sales volume based on advertising expenditures across TV, Radio, and Newspaper.

*   **Dataset:** ISLR Advertising Dataset (Spend vs Sales units).
*   **Approach:**
    *   **Baseline:** Linear Regression.
    *   **Final Model:** Random Forest Regressor (~98% accuracy).
*   **Sample Data:**
    *   `TV Ad Budget`: $150k
    *   `Radio Ad Budget`: $25k
    *   `Newspaper Ad Budget`: $30k
*   **How it Works:** The model identifies the **Return on Investment (ROI)** for each channel. It learned that TV advertising drives the vast majority of sales, while Newspaper advertising has a negligible impact. By manipulating budget sliders, the user can see an instant projection of "Sales Units" converted into estimated revenue.
*   **Aesthetic:** "Sunset Analytics" (Vibrant gradient dashboard with revenue gauges).

---

## 5. Credit Card Fraud Protection
**Goal:** Identify fraudulent transactions in a massive, highly imbalanced dataset.

*   **Dataset:** `mlg-ulb/creditcardfraud` (PCA-transformed European transactions).
*   **Approach:**
    *   **Imbalance Handling:** Utilized `RandomOverSampler` to balance the training distribution.
    *   **Model:** Random Forest Classifier (~89% F1-score on simulated data).
*   **Sample Data:**
    *   `V1` (Principal Vector): -1.35
    *   `V14` (Risk Vector): -4.21
    *   `Amount`: $800.00
*   **How it Works:** The model scans the Principal Component vectors (V1-V28) to find anomalous signatures. It triggers a "CRITICAL ALERT" if the transaction profile deviates significantly from secure historical patterns, providing a real-time risk probability score.
*   **Aesthetic:** "Cyber-Sentinel" (Dark tactical terminal with neon indicators).

---
*Developed by Bhagesh Biradar*
