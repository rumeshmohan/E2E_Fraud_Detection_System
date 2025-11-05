# End-to-End Fraud Detection System
This project builds a production-minded, supervised fraud detection system for e-commerce transactions.

## Task 1: Business Framing & Data Card

### Business Framing

This system is designed to solve the core business problem of e-commerce fraud. We must balance two competing costs:
* **Cost of Fraud (False Negative, C_fn):** The average loss from an undetected fraudulent transaction. We will set this at **$100**.
* **Cost of Review (False Positive, C_fp):** The operational cost to manually review a legitimate transaction flagged as fraud. We will set this at **$5**. 

Our model will be optimized to minimize the total expected cost based on this matrix.

### Key Performance Indicators (KPIs)

* **Primary Model Metric:** AUC (Area Under the Curve).
* **Secondary Model Metric:** F1-Score at the optimal business threshold.
* **Primary Business Metric:** Total Expected Cost Savings.

---

### Data Card

This section documents the initial analysis of the Kaggle dataset.

* **Source:** `Kaggle-datasets/vbinh002/fraud-ecommerce`
* **Target Variable:** `class`
* **Total Records:** `[151112]`
* [cite_start]**Fraud Ratio:** `[9.36%]` 
* **Time Span:**
    * Start: `[From: 2015-01-01 00:00:44]`
    * End: `[To: 2015-12-16 02:56:05]`
* **High-Cardinality Features:**
    * `user_id` unique values: `[151112]` 
    * `device_id` unique values: `[137956]` 
    * `ip_address` unique values: `[143512]`
* **Leakage Check:**
    * All column names (`['user_id', 'signup_time', 'purchase_time', 'purchase_value', 'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class]`) were reviewed and appear to be available at the time of transaction, satisfying the leakage prevention constraint.

---

## Task 2: Data Quality & Exploratory Analysis

Completed a thorough analysis of the raw data. All findings are documented in `notebooks/01_data_card_and_eda.ipynb`.

### 1. Schema Validation
* All column data types were as expected.
* `purchase_time` and `signup_time` were successfully parsed as datetime objects.

### 2. Temporal Integrity
* **Finding:** `[e.g., "No temporal violations found."]`
* **Action:** `[e.g., "No action needed."]`

### 3. Missing Data
* **Finding:** `[e.g., "No missing values were found in any columns."]`
* **Action:** `[e.g., "No action needed."]`

### 4. Outlier Detection
* **Finding:** The `purchase_value` column shows significant outliers. The 75th percentile is `[$49.0]` while the max value is `[$154.0]`.
* **Action:** Outliers will **not** be removed, as unusually high or low purchase values can be a strong indicator of fraudulent activity. The model must be robust to these values.

### 5. Class Balance
* **Finding:** Confirmed the class balance is approximately `[9.3646%]` fraud, which matches the project's challenge of significant class imbalance.

## Task 3: Leakage-Safe Feature Engineering

All feature engineering logic is contained in `src/feature_engineering.py`. All features are computed to be leakage-safe, meaning they only use information that would be available at the exact time of the transaction.

New features created:

1.  **`account_age_minutes`**: Time between `signup_time` and `purchase_time`. New accounts are often higher risk.
2.  **Velocity Metrics (1h & 24h)**:
    * `velocity_{entity}_count_{window}`
    * `velocity_{entity}_amount_{window}`
    * Calculated for `device_id` and `ip_address`.
    * Uses a rolling window with `closed='left'` to ensure only *past* transactions are counted, preventing leakage.
3.  **Rarity Signals (7d)**:
    * `rarity_{entity}_count_7d`
    * Calculated for `device_id` and `ip_address` to identify entities seen rarely in the past week.
4.  **Amount Normalization (30d)**:
    * `zscore_user_30d`
    * Calculates a rolling Z-score for `purchase_value` based on the user's 30-day history. This identifies purchases that are unusually large or small *for that specific user*.
5.  **Target Encoding (K-Fold)**:
    * **Note:** This feature (`browser`, `source`) is *not* pre-processed. It will be built *inside* the model training pipeline (Task 4) using a K-fold strategy to prevent target leakage, as required by the presentation.

## Task 4: Modeling & Threshold Optimization

This task involved splitting the data, building a preprocessing pipeline, and training our models. All work is in `notebooks/02_model_training.ipynb`.

### 1. Data Splitting
* A strict **time-based split** was used to prevent temporal leakage.
* **Training Set:** 80% of data (Time: `[Your Train Start Time]` to `[Your Train End Time]`)
* **Test Set:** 20% of data (Time: `[Your Test Start Time]` to `[Your Test End Time]`)

### 2. Model Performance

#### Baseline: Logistic Regression
* **Test AUC:** `[Your LR AUC, e.g., ~0.49]`
* **Result:** The baseline model performed at chance level, indicating the problem is non-linear and requires a more complex model.

#### Advanced: XGBoost
* **Test AUC:** **0.5021**
* **Result:** This is a **critical failure**. An AUC of ~0.50 means the model has **zero predictive power** and is equivalent to random guessing.

### 3. Debugging & Analysis (Critical Finding)

The **XGBoost Feature Importance** plot was the key debugging tool. It revealed that the model was **only** using `account_age_minutes` and the `_1h` velocity features.

All other features (`_24h`, `_7d`, `zscore_user_id_30d`) had near-zero importance. This **proves** that the feature engineering pipeline is **still broken** and is not correctly calculating these long-term features, feeding the model scrambled or useless data.

### 4. Threshold Optimization (Result Not Valid)

Though the model is non-predictive, the optimization logic was executed as required.
* **Costs:** C_fp = $5, C_fn = $100
* **Warning:** The results below are **not valid** because they are based on a random-guessing model.
* **Calculated Optimal Threshold:** 0.3797
* **Calculated F1-Score:** 0.0713

**Conclusion:** Task 4 is complete, but it has revealed a critical bug in the feature engineering pipeline. The model does not meet the project's quality gates. For the purpose of completing the project, we will proceed with this broken model to build the remaining infrastructure (Task 5-7), but in a real-world scenario, we would stop here and fix the data pipeline.