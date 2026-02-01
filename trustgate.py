# =========================
# TrustGate Benchmarking Framework
# FD001 vs FD002 vs FD003 vs FD004
# Multi-model, Colab-ready
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# -------------------------
# Data Loader
# -------------------------
def load_dataset(path):
    cols = ['engine_id', 'cycle'] + \
           [f'op_{i}' for i in range(1,4)] + \
           [f's{i}' for i in range(1,22)]
    return pd.read_csv(path, sep=' ', header=None).iloc[:, :-2].set_axis(cols, axis=1)

datasets = {
    'FD001': '/content/train_FD001.txt',
    'FD002': '/content/train_FD002.txt',
    'FD003': '/content/train_FD003.txt',
    'FD004': '/content/train_FD004.txt'
}

# -------------------------
# Add RUL Labels
# -------------------------
def add_rul(df):
    max_cycle = df.groupby('engine_id')['cycle'].max()
    df = df.merge(max_cycle, on='engine_id', suffixes=('', '_max'))
    df['RUL'] = df['cycle_max'] - df['cycle']
    return df.drop(columns=['cycle_max'])

# -------------------------
# TrustGate Evaluation
# -------------------------
def train_eval(df, model, model_name):
    X = df.drop(columns=['engine_id', 'cycle', 'RUL'])
    y = df['RUL']
    X = StandardScaler().fit_transform(X)

    model.fit(X, y)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    return model_name, mae

# -------------------------
# Models to Compare
# -------------------------
models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ("XGBoost", XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1))
]

# -------------------------
# Run TrustGate Benchmark
# -------------------------
results = []

for name, path in datasets.items():
    df = load_dataset(path)
    df = add_rul(df)
    for model_name, model in models:
        mname, mae = train_eval(df, model, model_name)
        results.append([name, mname, mae])

results_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'MAE (cycles)'])
print(results_df)

# -------------------------
# Visualization
# -------------------------
plt.figure(figsize=(10,6))
for model_name in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model_name]
    plt.plot(subset['Dataset'], subset['MAE (cycles)'], marker='o', label=model_name)

plt.xlabel('C-MAPSS Subset')
plt.ylabel('MAE (cycles)')
plt.title('TrustGate Benchmark: RUL Prediction Difficulty (FD001–FD004)')
plt.legend()
plt.grid(True)
plt.show()
