# src/evaluation.py
# =========================================================
# Part 4: Evaluation & Visual Analytics
# =========================================================
# - NASA official score (TEST set)
# - Global model performance visualizations
# - Adaptive fusion analysis
# - Risk score distribution
# - SHAP explainability (sensor modality, validation set)
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

sns.set(style="whitegrid")


# =========================================================
# NASA OFFICIAL SCORE
# =========================================================
def nasa_score(y_true, y_pred):
    """
    NASA C-MAPSS asymmetric scoring function.
    Late predictions are penalized more than early ones.
    """
    d = y_pred - y_true
    score = 0.0
    for di in d:
        if di < 0:   # early (safe)
            score += np.exp(-di / 13) - 1
        else:        # late (dangerous)
            score += np.exp(di / 10) - 1
    return score


# =========================================================
# MAIN EVALUATION FUNCTION (EXACT NOTEBOOK LOGIC)
# =========================================================
def run_evaluation(
    y_test,
    y_pred_fused,
    y_val,
    val_predictions,
    MODALITIES,
    risk_true,
    risk_pred,
    X_val,
    ALL_FEATURES,
    modality_models
):
    # ========================================
    # FINAL TEST METRICS
    # ========================================
    mae_test = mean_absolute_error(y_test, y_pred_fused)
    nasa_test = nasa_score(y_test, y_pred_fused)

    print("🎯 FINAL TEST PERFORMANCE (GLOBAL MODEL)")
    print(f"   MAE:        {mae_test:.2f} cycles")
    print(f"   NASA Score: {nasa_test:,.0f}")
    print(f"   Risk range: [{risk_pred.min():.3f}, {risk_pred.max():.3f}]")


    # ========================================
    # VISUAL 1: TRUE vs PREDICTED RUL
    # ========================================
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred_fused, alpha=0.5, s=15)
    plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL (Global Multimodal Model)")
    plt.show()


    # ========================================
    # VISUAL 2: RESIDUAL DISTRIBUTION
    # ========================================
    residuals = y_pred_fused - y_test

    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40, alpha=0.8)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Prediction Error (Predicted − True RUL)")
    plt.ylabel("Frequency")
    plt.title("Residual Error Distribution (Test Set)")
    plt.show()


    # ========================================
    # VISUAL 3: ERROR vs TRUE RUL (BIAS CHECK)
    # ========================================
    plt.figure(figsize=(7, 4))
    plt.scatter(y_test, residuals, alpha=0.5, s=15)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("True RUL")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error vs True RUL")
    plt.show()


    # ========================================
    # VISUAL 4: MODALITY PERFORMANCE (VALIDATION)
    # ========================================
    modality_mae = {
        modality: mean_absolute_error(y_val, val_predictions[modality])
        for modality in MODALITIES
    }

    plt.figure(figsize=(6, 4))
    plt.bar(modality_mae.keys(), modality_mae.values())
    plt.ylabel("MAE (cycles)")
    plt.title("Validation MAE per Modality")
    plt.show()


    # ========================================
    # VISUAL 5: ADAPTIVE FUSION TRUST WEIGHTS
    # ========================================
    reconstructed_weights = {}
    total_weight = 0.0

    for modality in MODALITIES:
        r2 = r2_score(y_val, val_predictions[modality])
        weight_val = max(0.0, r2)
        reconstructed_weights[modality] = weight_val
        total_weight += weight_val

    if total_weight == 0:
        reconstructed_weights = {m: 1.0 for m in MODALITIES}

    plt.figure(figsize=(6, 4))
    plt.bar(reconstructed_weights.keys(), reconstructed_weights.values())
    plt.ylabel("Fusion Weight (Trust)")
    plt.title("Adaptive Modality Trust Weights (R²-based)")
    plt.show()


    # ========================================
    # VISUAL 6: RISK SCORE DISTRIBUTION
    # ========================================
    plt.figure(figsize=(7, 4))
    plt.hist(risk_true, bins=30, alpha=0.6, label="True Risk")
    plt.hist(risk_pred, bins=30, alpha=0.6, label="Predicted Risk")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.title("Risk Score Distribution (Test Set)")
    plt.legend()
    plt.show()


    # ========================================
    # SHAP EXPLAINABILITY (SENSOR MODALITY)
    # ========================================
    print("\n🔍 SHAP Explainability (Sensor Modality – Validation Set)")

    try:
        import shap

        sample_idx = np.random.choice(
            len(X_val), size=min(100, len(X_val)), replace=False
        )

        sensor_indices = [
            ALL_FEATURES.index(col) for col in MODALITIES['sensor']
        ]

        X_sample = pd.DataFrame(
            X_val[sample_idx][:, sensor_indices],
            columns=MODALITIES['sensor']
        )

        explainer = shap.TreeExplainer(modality_models['sensor'])
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Summary Plot – Sensor Modality")
        plt.show()

    except ImportError:
        print("SHAP not installed. Run: pip install shap")
    except Exception as e:
        print(f"SHAP error: {e}")


    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n✅ PART 4 COMPLETED SUCCESSFULLY")
    print("   • Global multimodal RUL regression")
    print("   • Adaptive R²-weighted fusion")
    print("   • NASA official evaluation metric")
    print("   • Risk-aware decision support")
    print("   • Explainability via SHAP")
