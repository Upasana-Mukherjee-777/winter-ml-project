import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def add_rolling_features(df, sensors, window):
    df_new = df.copy()
    for s in sensors:
        df_new[f"{s}_roll_mean_w{window}"] = (
            df_new.groupby("engine_id")[s]
            .rolling(window, min_periods=1)
            .mean()
            .values
        )
        df_new[f"{s}_roll_std_w{window}"] = (
            df_new.groupby("engine_id")[s]
            .rolling(window, min_periods=1)
            .std()
            .fillna(0)
            .values
        )
    return df_new


def evaluate_rolling_windows(
    train_df,
    useful_sensors,
    sensor_features,
    train_engines,
    val_engines,
    window_sizes,
    n_estimators=50,
    max_depth=12
):
    results = []

    for w in window_sizes:
        df_w = add_rolling_features(train_df, useful_sensors, w)
        rolling_features = [c for c in df_w.columns if f"_w{w}" in c]
        feature_cols = sensor_features + rolling_features

        X = df_w[feature_cols].values
        y = df_w["RUL"].values
        engines = df_w["engine_id"].values

        train_mask = np.isin(engines, train_engines)
        val_mask = np.isin(engines, val_engines)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X[train_mask], y[train_mask])
        preds = rf.predict(X[val_mask])

        mae = mean_absolute_error(y[val_mask], preds)
        results.append([w, mae])

    return pd.DataFrame(
        results,
        columns=["Rolling Window Size", "MAE (Val)"]
    )
