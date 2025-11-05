# End-to-End Fraud Detection System

This project builds a production-minded, supervised fraud detection system for e-commerce transactions. It uses a public Kaggle dataset and focuses on real-world deployment considerations like data quality, leakage prevention, and cost-aware modeling.

---

## 📌 Task 1: Business Framing & Cost Matrix

### Business Problem
E-commerce platforms face a financial trade-off:
* **Fraud Losses:** Cost incurred from undetected fraudulent transactions (False Negatives).
* **Operational Costs:** Cost of manually reviewing transactions flagged as fraud (False Positives).

This model's goal is to find an optimal threshold that minimizes the total cost by balancing these two competing objectives.

### Cost Matrix
To optimize the model for business impact, we will use the following assumed costs:
* **`C_fp` (False Positive Cost): $5** - The cost of a human analyst manually reviewing a legitimate transaction.
* **`C_fn` (False Negative Cost): $100** - The average financial loss from an undetected fraudulent transaction. (This is a chosen value from the provided range of $50-$200).

### Key Performance Indicators (KPIs)
The model will be evaluated based on:
1.  **AUC-ROC:** Primary metric for model discrimination.
2.  **Expected Cost Savings:** Total cost saved by the model at its optimal threshold, compared to a baseline.
3.  **Fraud Detection Rate (Recall):** Percentage of all fraud cases successfully caught.
4.  **Alert Precision:** Percentage of fraud alerts that are *actually* fraud.

---

## 📊 Project Status

### ✅ Phase 1: Foundation & Data Engineering (Complete)

This phase covers all data acquisition, validation, and feature engineering, resulting in a model-ready "curated" dataset.

* **Task 1: Business Framing** 
    * The business problem is framed as a cost-minimization task.
    * **Cost Matrix (Assumed):**
        * **`C_fp` (False Positive Cost): $5** (cost of manual review) 
        * **`C_fn` (False Negative Cost): $100** (avg. fraud loss) 
    * **Primary Metric:** AUC-ROC
    * **Primary Business KPI:** Minimized Total Cost / Expected Cost Savings 

* **Task 2: Data Quality & EDA** 
    * Analysis is documented in `notebooks/01-eda-and-validation.ipynb`.
    * **Row Count:** 151,112 transactions confirmed.
    * **Class Balance:** 9.36% fraud, 90.64% not-fraud (matches ~10% expectation).
    * **Data Quality:**
        * **Missing Data:** 0% missing values across all columns.
        * **Temporal Integrity:** 100% of records pass (`purchase_time >= signup_time`).

* **Task 3: Leakage-Safe Feature Engineering**
    * **Core Logic:** Implemented in `src/feature_engineering.py`.
    * **Leakage Prevention:** All rolling-window features are 100% leakage-safe, using `closed='left'` to ensure only historical data is used for each transaction.
    * **Features Created:**
        * **`account_age_minutes`**: Time between signup and purchase.
        * **Velocity Metrics**: Transaction counts and amounts for `device_id` & `ip_address` (1h, 24h windows).
        * **Rarity Signals**: Transaction counts for `device_id` & `ip_address` (7d window).
        * **Amount Normalization**: Z-score of `purchase_value` for `user_id` (30d window).
    * **Pipeline Script:** `src/run_data_processing.py` orchestrates the full pipeline.
    * **Output Artifact:** A `data/curated_dataset.parquet` file is generated, ready for model training.