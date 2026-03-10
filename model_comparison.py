# =========================================================
# Model Comparison & Ablation Studies
# =========================================================
# - Compares multiple regression models
# - Uses MAE, R², and NASA Score
# - NumPy-safe (no .values usage)
# =========================================================

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# Optional XGBoost
# ---------------------------------------------------------
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------
# NASA SCORE (Official C-MAPSS Metric)
# ---------------------------------------------------------
def nasa_score(y_true, y_pred):
    """
    NASA asymmetric scoring function.
    Late predictions are penalized more than early ones.
    """
    score = 0.0
    for yt, yp in zip(y_true, y_pred):
        d = yp - yt
        if d < 0:   # early (safe)
            score += np.exp(-d / 13) - 1
        else:       # late (dangerous)
            score += np.exp(d / 10) - 1
    return score


# ---------------------------------------------------------
# REGRESSION MODEL ZOO
# ---------------------------------------------------------
def get_regression_models(random_state=42):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="reg:squarederror"
        )

    return models


# ---------------------------------------------------------
# MODEL COMPARISON (VALIDATION SET)
# ---------------------------------------------------------
def compare_models(X_train, y_train, X_val, y_val, random_state=42):
    """
    Trains multiple regression models and compares them
    using validation MAE, R², and NASA Score.
    """
    models = get_regression_models(random_state)
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        results.append({
            "Model": name,
            "MAE (Val)": mean_absolute_error(y_val, preds),
            "R² (Val)": r2_score(y_val, preds),
            "NASA Score (Val)": nasa_score(y_val, preds)
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("MAE (Val)").reset_index(drop=True)
    return results_df, trained_models


# ---------------------------------------------------------
# VOTING ENSEMBLE
# ---------------------------------------------------------
def build_voting_ensemble(trained_models, model_names, X_train, y_train):
    """
    Builds and trains a VotingRegressor from selected models.
    """
    estimators = [(name, trained_models[name]) for name in model_names]
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(X_train, y_train)
    return ensemble


# ---------------------------------------------------------
# FINAL TEST EVALUATION
# ---------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on the test set.
    """
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "R²": r2_score(y_test, preds),
        "NASA Score": nasa_score(y_test, preds)
    }
