import pandas as pd
import numpy as np
import os
import sys
import mlflow

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler # <-- Re-added StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression # <-- NEW: Import baseline model
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# --- Config ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CURATED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'curated_dataset.parquet')
MLFLOW_EXPERIMENT_NAME = "Fraud_Detection_System"

COST_FP = 5
COST_FN = 100
TARGET_COLUMN = 'class'

def load_data():
    """Loads the curated feature dataset."""
    print(f"Loading curated data from {CURATED_DATA_PATH}...")
    try:
        df = pd.read_parquet(CURATED_DATA_PATH)
        df = df.sort_values('purchase_time').reset_index(drop=True)
        print(f"Data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Curated data file not found at {CURATED_DATA_PATH}")
        sys.exit(1)

def get_features_and_target(df):
    """Separates features (X) and target (y)."""
    excluded_cols = [
        TARGET_COLUMN, 'user_id', 'device_id', 'ip_address',
        'signup_time', 'purchase_time', 'unique_row_id'
    ]
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    
    categorical_features = ['source', 'browser', 'sex']
    numerical_features = [col for col in feature_cols if col not in categorical_features]
    
    return X, y, categorical_features, numerical_features

def calculate_total_cost(y_true, y_pred):
    """Calculates total business cost."""
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    total_cost = (fp * COST_FP) + (fn * COST_FN)
    return total_cost

def find_optimal_threshold(model, X_val, y_val):
    """Finds the probability threshold that minimizes total business cost."""
    print("  Optimizing threshold for cost...")
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    
    min_cost = float('inf')
    optimal_threshold = 0.5
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        cost = calculate_total_cost(y_val, y_pred)
        if cost < min_cost:
            min_cost = cost
            optimal_threshold = t
            
    print(f"  Optimal threshold found: {optimal_threshold:.4f} (Min Cost: ${min_cost})")
    return optimal_threshold, min_cost

def main():
    """Main training and evaluation function."""
    
    df = load_data()
    X, y, categorical_features, numerical_features = get_features_and_target(df)
    
    # --- Time-Based Split ---
    n_rows = len(df)
    train_val_split_idx = int(n_rows * 0.85)
    
    X_train_val = X.iloc[:train_val_split_idx]
    y_train_val = y.iloc[:train_val_split_idx]
    X_test = X.iloc[train_val_split_idx:]
    y_test = y.iloc[train_val_split_idx:]
    
    print(f"Data split: Train/Val ({len(X_train_val)} rows), Test ({len(X_test)} rows)")
    
    # --- MLflow Setup ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # --- NEW: Define Preprocessing Pipelines ---

    # Preprocessor for LINEAR models (Logistic Regression)
    # This scales numerical data and one-hot encodes categoricals
    linear_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # Preprocessor for TREE models (XGBoost)
    # This passes numerical data through and one-hot encodes categoricals
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # --- NEW: Define Model Configurations ---
    model_configs = [
        {
            "name": "Baseline_LogisticRegression",
            "pipeline": Pipeline([
                ('preprocessor', linear_preprocessor),
                ('model', LogisticRegression(
                    class_weight='balanced', # Handles imbalance for LR
                    max_iter=1000
                ))
            ]),
            "model_params": {}
        },
        {
            "name": "Advanced_XGBoost",
            "pipeline": Pipeline([
                ('preprocessor', tree_preprocessor),
                ('model', XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc'
                    # scale_pos_weight set dynamically in loop
                ))
            ]),
            "model_params": {} # We set scale_pos_weight in the loop
        }
    ]

    best_test_auc = -1.0
    best_model_name = ""

    # Parent MLflow run to group the two models
    with mlflow.start_run(run_name="Model_Comparison_Run") as parent_run:
        print(f"Starting Parent MLflow Run: {parent_run.info.run_id}")
        mlflow.log_params({
            "cost_fp": COST_FP,
            "cost_fn": COST_FN,
            "time_split_ratio": 0.85
        })
        
        # --- Loop through each model config ---
        for config in model_configs:
            model_name = config["name"]
            pipeline = config["pipeline"]
            
            # Start a nested run for this specific model
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                print(f"\n--- Starting training for: {model_name} ---")
                
                # --- Time-Series Cross-Validation ---
                tscv = TimeSeriesSplit(n_splits=5)
                val_aucs = []
                
                print("  Starting time-series cross-validation...")
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
                    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
                    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
                    
                    # Set model-specific params (like scale_pos_weight for XGB)
                    if "XGBClassifier" in pipeline.named_steps['model'].__class__.__name__:
                        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
                        pipeline.named_steps['model'].set_params(scale_pos_weight=scale_weight)
                    
                    pipeline.fit(X_train, y_train)
                    y_val_probs = pipeline.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, y_val_probs)
                    val_aucs.append(auc)
                    print(f"    Fold {fold+1}/5 AUC: {auc:.4f}")
                
                avg_val_auc = np.mean(val_aucs)
                print(f"  Average Validation AUC: {avg_val_auc:.4f}")
                mlflow.log_metric("validation_auc_avg", avg_val_auc)
                
                # --- Final Training & Thresholding ---
                print("  Training final model on all Train/Val data...")
                # Set final scale_pos_weight
                if "XGBClassifier" in pipeline.named_steps['model'].__class__.__name__:
                    scale_weight = (y_train_val == 0).sum() / (y_train_val == 1).sum()
                    pipeline.named_steps['model'].set_params(scale_pos_weight=scale_weight)
                
                pipeline.fit(X_train_val, y_train_val)
                
                optimal_threshold, test_cost = find_optimal_threshold(pipeline, X_test, y_test)
                mlflow.log_metric("optimal_threshold", optimal_threshold)
                mlflow.log_metric("test_set_cost", test_cost)
                
                # --- Evaluate on Test Set ---
                y_test_probs = pipeline.predict_proba(X_test)[:, 1]
                y_test_pred_optimal = (y_test_probs >= optimal_threshold).astype(int)
                
                test_auc = roc_auc_score(y_test, y_test_probs)
                test_f1 = f1_score(y_test, y_test_pred_optimal)
                test_precision = precision_score(y_test, y_test_pred_optimal)
                test_recall = recall_score(y_test, y_test_pred_optimal)
                
                print(f"\n  --- Test Set Performance ({model_name}) ---")
                print(f"  Test AUC: {test_auc:.4f}")
                print(f"  Test F1-score: {test_f1:.4f}")
                print(f"  Test Precision: {test_precision:.4f}")
                print(f"  Test Recall: {test_recall:.4f}")
                
                mlflow.log_metrics({
                    "test_auc": test_auc,
                    "test_f1_optimal": test_f1,
                    "test_precision_optimal": test_precision,
                    "test_recall_optimal": test_recall
                })
                
                # Log the model
                print("  Logging model to MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name=f"{model_name}_model",
                    input_example=X_train_val.head()
                )
                
                # --- Check if this is the best model ---
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_model_name = model_name

        # --- MODEL PERFORMANCE GATE (Task 6) ---
        # Now happens *after* all models run, using the best score
        print("\n--- Checking Model Performance Gate (AUC >= 0.5) ---")
        AUC_GATE = 0.5
        
        if best_test_auc < AUC_GATE:
            print(f"!!!!!!!!!!!!!! MODEL FAILED PERFORMANCE GATE !!!!!!!!!!!!!!")
            print(f"Best Test AUC ({best_model_name}: {best_test_auc:.4f}) is below the required gate ({AUC_GATE}).")
            print(f"Build will fail.")
            sys.exit(1)
        
        print(f"SUCCESS: Model passed performance gate (Best Model: {best_model_name}, Test AUC: {best_test_auc:.4f}).")
        
        # Log the best model to the parent run for easy access
        mlflow.log_metric("best_model_auc", best_test_auc)
        mlflow.log_param("best_model_name", best_model_name)
        print("Model training pipeline finished.")
        
if __name__ == "__main__":
    main()