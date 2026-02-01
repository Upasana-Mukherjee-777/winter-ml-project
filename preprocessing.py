import numpy as np
import pandas as pd

# ---------------------------------------------------------
# FD002 SENSOR FILTERING
# ---------------------------------------------------------
def get_useful_sensors(train_df):
    invalid_sensors = ['sensor_20', 'sensor_21']

    candidate_sensors = [
        f'sensor_{i}' for i in range(1, 20)
        if f'sensor_{i}' not in invalid_sensors
    ]

    sensor_var = train_df[candidate_sensors].var()
    low_var_sensors = sensor_var[sensor_var < 0.01].index.tolist()

    useful_sensors = [
        s for s in candidate_sensors if s not in low_var_sensors
    ]

    meta = {
        "invalid_sensors": invalid_sensors,
        "low_variance_sensors": low_var_sensors
    }

    return useful_sensors, meta


# ---------------------------------------------------------
# RUL COMPUTATION (FULL RANGE)
# ---------------------------------------------------------
def compute_rul_full(df):
    df = df.copy()
    df["RUL"] = (
        df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    )
    return df


# ---------------------------------------------------------
# 3-SIGMA OUTLIER CAPPING
# ---------------------------------------------------------
def apply_3sigma_capping(df, sensor_list):
    df = df.copy()
    cap_report = {}

    for s in sensor_list:
        mean = df[s].mean()
        std = df[s].std()

        lower = mean - 3 * std
        upper = mean + 3 * std

        outliers = ((df[s] < lower) | (df[s] > upper)).sum()
        df[s] = df[s].clip(lower, upper)

        cap_report[s] = int(outliers)

    return df, cap_report
