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