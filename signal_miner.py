import numpy as np
import pandas as pd
import random
import lightgbm as lgb

# Randomized parameter grid
def get_ran_cfg(param_dict):
    return {k: random.sample(v, 1)[0] for k, v in param_dict.items()}

def get_rdn_cfgs(param_dict, num):
    configurations = []
    while len(configurations) < num:
        cfg = get_ran_cfg(param_dict)
        if (cfg not in configurations) and (cfg['num_leaves'] <= 2 ** cfg['max_depth']):
            configurations.append(cfg)
    return sorted(configurations, key=lambda d: d['max_depth'])

def get_model(cfg):
    model = lgb.LGBMRegressor(
        colsample_bytree=cfg['colsample_bytree'],
        max_bin=cfg['max_bin'],
        max_depth=cfg['max_depth'],
        num_leaves=cfg['num_leaves'],
        min_child_samples=cfg['min_child_samples'],
        n_estimators=cfg['n_estimators'],
        reg_lambda=cfg['reg_lambda'],
        learning_rate=cfg['learning_rate'],
        n_jobs=cfg['n_jobs']
    )
    
    return model
    

def evaluate_completed_configs(data, configurations, mmapped_array, done_splits, all_splits, ns, label='target'):
    """
    Evaluate completed configurations and return a DataFrame with evaluation metrics.

    Parameters:
        data (pd.DataFrame): The dataset with eras, features, and mmap_idx.
        configurations (list): List of configuration dictionaries.
        mmapped_array (np.memmap): Memory-mapped array for predictions.
        done_splits (np.memmap): Memory-mapped array tracking completed splits.
        all_splits (list): List of validation-test split indices.
        ns (int): Number of splits per configuration.
        label (str): Target column name.

    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for completed configurations.
    """
    # Identify completed configurations
    done_configs = [k for k in range(len(configurations)) if np.sum(done_splits[k * ns:k * ns + ns]) == ns]

    # Extract validation and test eras
    validation_first_date = all_splits[0][1][0]
    validation_last_date = all_splits[0][1][-1]
    test_first_date = all_splits[-1][1][0]

    eval_validation = data.loc[(data['era'] >= validation_first_date) & (data['era'] <= validation_last_date), [label, 'era', 'mmap_idx']].copy().dropna(subset=[label])
    eval_test = data.loc[data['era'] >= test_first_date, [label, 'era', 'mmap_idx']].copy().dropna(subset=[label])
    eval_whole = data.loc[data['era'] >= validation_first_date, [label, 'era', 'mmap_idx']].copy().dropna(subset=[label])

    validation_stats = {'corr': [], 'corr_shp': [], 'max_dd': []}
    test_stats = {'corr': [], 'corr_shp': [], 'max_dd': []}
    whole_stats = {'corr': [], 'corr_shp': [], 'max_dd': []}

    eval_validation_idx = eval_validation['mmap_idx'].values
    eval_test_idx = eval_test['mmap_idx'].values
    eval_whole_idx = eval_whole['mmap_idx'].values

    # Evaluate each completed configuration
    for i in done_configs:
        eval_validation['pred'] = mmapped_array[eval_validation_idx, i]
        validation_era_results = eval_validation.groupby('era')[[label, 'pred']].apply(lambda x: x[[label, 'pred']].dropna().corr().iloc[0, 1]).values
        
        cumpnl = np.nancumsum(validation_era_results)
        cummax = np.maximum.accumulate(cumpnl)
        max_dd = np.max(cummax - cumpnl)
        
        validation_stats['corr'].append(np.nanmean(validation_era_results))
        validation_stats['corr_shp'].append(np.nanmean(validation_era_results) / np.nanstd(validation_era_results))
        validation_stats['max_dd'].append(max_dd)

        eval_test['pred'] = mmapped_array[eval_test_idx, i]
        test_era_results = eval_test.groupby('era')[[label, 'pred']].apply(lambda x: x[[label, 'pred']].dropna().corr().iloc[0, 1]).values

        cumpnl = np.nancumsum(test_era_results)
        cummax = np.maximum.accumulate(cumpnl)
        max_dd = np.max(cummax - cumpnl)

        test_stats['corr'].append(np.nanmean(test_era_results))
        test_stats['corr_shp'].append(np.nanmean(test_era_results) / np.nanstd(test_era_results))
        test_stats['max_dd'].append(max_dd)

        eval_whole['pred'] = mmapped_array[eval_whole_idx, i]
        whole_era_results = eval_whole.groupby('era')[[label, 'pred']].apply(lambda x: x[[label, 'pred']].dropna().corr().iloc[0, 1]).values

        cumpnl = np.nancumsum(whole_era_results)
        cummax = np.maximum.accumulate(cumpnl)
        max_dd = np.max(cummax - cumpnl)

        whole_stats['corr'].append(np.nanmean(whole_era_results))
        whole_stats['corr_shp'].append(np.nanmean(whole_era_results) / np.nanstd(whole_era_results))
        whole_stats['max_dd'].append(max_dd)

    # Combine results into a DataFrame
    config_df = pd.concat([
        pd.DataFrame(configurations).iloc[done_configs],
        pd.DataFrame(validation_stats, index=done_configs),
        pd.DataFrame(test_stats, index=done_configs),
        pd.DataFrame(whole_stats, index=done_configs)
    ], axis=1)

    config_df.columns = list(config_df.columns[:-9]) + [
        'validation_corr', 'validation_shp', 'validation_max_dd',
        'test_corr', 'test_shp', 'test_max_dd',
        'whole_corr', 'whole_shp', 'whole_max_dd'
    ]

    return config_df.sort_values('validation_shp').dropna()

def compare_to_benchmark(res_df, benchmark_id=0):
    """
    Compare random configurations to the benchmark.

    Parameters:
        res_df (pd.DataFrame): DataFrame with evaluation results for all configurations.
        benchmark_id (int): Index of the benchmark configuration in res_df.

    Returns:
        pd.DataFrame: Subset of res_df where random configurations outperform the benchmark.
    """
    # Extract benchmark values
    benchmark_corr = res_df.loc[benchmark_id, 'whole_corr']
    benchmark_sharpe = res_df.loc[benchmark_id, 'whole_shp']

    # Find configurations that beat the benchmark
    outperforming_configs = res_df[
        (res_df['whole_corr'] > benchmark_corr) & 
        (res_df['whole_shp'] > benchmark_sharpe) & 
        (res_df.index != benchmark_id)  # Exclude the benchmark itself
    ]

    return outperforming_configs

def evaluate_and_ensemble(ensemble, configurations, mmapped_array, data, all_splits, feature_cols, get_model, save_name="model"):
    """
    Compare configurations to the benchmark, select the best, retrain it on all data,
    and package it into a predict function for deployment.

    Parameters:
        ensemble (list): List of configuration ids to retrain and export
        configurations (list): List of configuration dictionaries.
        mmapped_array (np.memmap): Memory-mapped array for predictions.
        data (pd.DataFrame): The dataset with eras, features, and mmap_idx.
        all_splits (list): List of train-test split indices.
        feature_cols (list): List of feature column names.
        get_model (function): Function to initialize the model based on configuration.
        save_name (str): Name to use for saving the pickle file.

    Returns:
        None
    """
    print(f"Selected ensemble: {ensemble}")

    # Step 3: Validate the model replicates last fold performance
    train_didxs, test_didxs = all_splits[-1]
    k = ensemble[0]

    cfg = configurations[k]
    label = cfg['target']
    train_rows = (data['era'].isin(np.array(sorted(data['era'].unique()))[train_didxs])) & (~data[label].isna())
    test_rows = (data['era'].isin(np.array(sorted(data['era'].unique()))[test_didxs])) & (~data[label].isna())

    model = get_model(cfg)
    model.fit(
        data.loc[train_rows, feature_cols].values,
        data.loc[train_rows, label].values
    )

    result_vector = model.predict(data.loc[test_rows, feature_cols].values)
    if not np.isclose(result_vector, mmapped_array[test_rows, k], rtol=1e-03, atol=1e-05).all():
        print("Model did not replicate last fold performance. Check your implementation.")
        return

    # Step 4: Retrain the ensemble on all data
    models = []
    for k in ensemble:
        cfg = configurations[k]
        label = cfg['target']
        train_rows = (~data[label].isna())

        model = get_model(cfg)
        model.fit(
            data.loc[train_rows, feature_cols].values,
            data.loc[train_rows, label].values
        )

        models.append(model)

    # Step 5: Define the predict function
    def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        i = 0
        for model in models:
            live_features[f'pred_temp'] = model.predict(live_features[feature_cols].values)
            live_features[f'pred_{i}_rank'] = live_features[f'pred_temp'].rank(pct=True)
            i += 1
        live_predictions = live_features[[f'pred_{i}_rank' for i in range(len(models))]].mean(axis=1)
        submission = pd.Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")

    # Step 6: Save the predict function
    import cloudpickle
    p = cloudpickle.dumps(predict)
    with open(f"predict_{save_name}_full.pkl", "wb") as f:
        f.write(p)

    print(f"Predict function saved as predict_{save_name}_full.pkl")
