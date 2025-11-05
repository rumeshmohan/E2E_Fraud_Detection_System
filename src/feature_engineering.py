import pandas as pd
import numpy as np

def compute_account_age(df_raw):
    """
    Computes account age. This is fast and can be run first.
    """
    print("Computing 'account_age_minutes'...")
    df = df_raw.copy()
    
    # Ensure datetime conversion
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    
    # Filter out any temporal violations
    valid_times = df['purchase_time'] >= df['signup_time']
    print(f"  Dropped {len(df) - valid_times.sum()} rows with temporal violations.")
    df = df[valid_times].copy()
    
    time_diff = df['purchase_time'] - df['signup_time']
    df['account_age_minutes'] = time_diff.dt.total_seconds() / 60
    
    # Sort by time and create the unique key for merging
    df = df.sort_values('purchase_time').reset_index(drop=True)
    df['unique_row_id'] = df.index
    
    return df

def compute_group_features(df_group, group_col, windows_metrics):
    """
    This is the helper function that will be applied to each group.
    """
    # group is a small DataFrame (e.g., all rows for one device_id)
    # It must be sorted by time for rolling to work
    group = df_group.set_index('purchase_time').sort_index()
    
    group['count'] = 1
    group['amount'] = group['purchase_value']
    
    for window, metric in windows_metrics:
        col_name = ""
        rolling_series = None
        
        if metric == 'count':
            rolling_series = group['count'].rolling(window, closed='left').sum()
            col_name = f'velocity_{group_col}_count_{window}'
            if window == '7d':
                col_name = f'rarity_{group_col}_count_{window}'
        
        elif metric == 'amount':
            rolling_series = group['amount'].rolling(window, closed='left').sum()
            col_name = f'velocity_{group_col}_amount_{window}'
            
        elif metric == 'zscore':
            rolling_mean = group['purchase_value'].rolling(window, closed='left').mean()
            rolling_std = group['purchase_value'].rolling(window, closed='left').std()
            zscore = (group['purchase_value'] - rolling_mean) / rolling_std
            col_name = f'zscore_{group_col}_{window}'
            group[col_name] = zscore
            continue
            
        if rolling_series is not None:
            group[col_name] = rolling_series
            
    # Reset index to get 'purchase_time' back
    group = group.reset_index()
    
    # Get all new column names we just created
    new_cols = [c for c in group.columns if c.startswith(('velocity_', 'rarity_', 'zscore_'))]
    
    # We ONLY need to return the new features and the merge key
    return group[['unique_row_id'] + new_cols]

def create_all_features(df_raw):
    """
    Main orchestrator function. Runs all feature engineering steps.
    """
    # 1. Compute account age (this also sorts and adds 'unique_row_id')
    df_main = compute_account_age(df_raw)
    
    # 2. Define feature jobs
    feature_jobs = [
        {
            'group_col': 'device_id',
            'windows_metrics': [('1h', 'count'), ('1h', 'amount'), ('24h', 'count'), ('24h', 'amount'), ('7d', 'count')]
        },
        {
            'group_col': 'ip_address',
            'windows_metrics': [('1h', 'count'), ('1h', 'amount'), ('24h', 'count'), ('24h', 'amount'), ('7d', 'count')]
        },
        {
            'group_col': 'user_id',
            'windows_metrics': [('30d', 'zscore')]
        }
    ]
    
    final_df = df_main.copy()
    
    # 3. Run all jobs
    for job in feature_jobs:
        group_col = job['group_col']
        windows_metrics = job['windows_metrics']
        
        print(f"Computing features for group: {group_col}...")
        
        # Group by the entity
        grouped_df = df_main.groupby(group_col)
        
        # Apply the helper function to each group
        df_features = grouped_df.apply(
            compute_group_features, 
            group_col=group_col, 
            windows_metrics=windows_metrics
        )
        
        # Clean the multi-index from the apply
        df_features = df_features.reset_index(drop=True)
        
        # 4. Merge results back to main df
        final_df = pd.merge(
            final_df,
            df_features,
            on='unique_row_id',
            how='left'
        )
        
        # De-duplicate columns if any
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # 5. Final Fillna
    feature_cols = [c for c in final_df.columns if c.startswith(('velocity_', 'rarity_', 'zscore_'))]
    print(f"Filling NaNs for {len(feature_cols)} new feature columns...")
    final_df[feature_cols] = final_df[feature_cols].fillna(0)
    
    # Z-score-specific: Handle inf/-inf from std=0
    final_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print("✅ Feature engineering complete.")
    return final_df