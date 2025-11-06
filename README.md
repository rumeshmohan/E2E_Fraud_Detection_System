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

---

### ✅ Phase 2: Batch Model Pipeline (Complete)

This phase focuses on training a predictive model on the curated data, optimizing it for business cost, and tracking all results.

* **Task 4: Modeling**
    * A training script `src/run_model_training.py` has been built to consume the artifact from Phase 1.
    * It correctly uses a time-based split (85% train/val, 15% test) to create a hold-out test set.
    * It uses `TimeSeriesSplit(n_splits=5)` for robust, leakage-safe cross-validation.
    * It uses an `XGBClassifier` with `scale_pos_weight` to handle class imbalance.
    * A cost-optimization function (`find_optimal_threshold`) is used to find the best probability threshold to minimize business costs.

* **Task 5: Experiment Tracking**
    * The training script is fully integrated with `mlflow`.
    * All runs are logged to the "Fraud_Detection_System" experiment.
    * **Metrics Logged:** `validation_auc_avg`, `optimal_threshold`, `test_set_cost`, `test_auc`, `test_f1_optimal`, `test_precision_optimal`, `test_recall_optimal`.
    * **Artifacts Logged:** The final, trained `scikit-learn` pipeline (including the preprocessor and model) is saved to MLflow.

* **Current Results**
    * **Average Validation AUC:** 0.4984
    * **Test AUC:** 0.5098
    * **Optimal Threshold:** 0.3466
    * **Test Set Cost:** $102,060
    * **Test F1-score:** 0.0803
    * **Test Precision:** 0.0509
    * **Test Recall:** 0.1896

* **Status:** The batch pipeline (Tasks 1-5) is now fully automated via the `Makefile`. All steps execute end-to-end. The current model's performance (`Test AUC: 0.5098`) does not meet the project's **Model Performance Gate (AUC >= 0.75)**. The next step is to iterate on the preprocessing and modeling logic to improve the AUC.