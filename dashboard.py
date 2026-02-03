# ============================
# DASHBOARD: DECISION SUPPORT
# ============================

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Risk conversion (FD002)
# ----------------------------
def rul_to_risk(rul, alpha=150, beta=20):
    return 1 / (1 + np.exp((rul - alpha) / beta))


# ----------------------------
# Main dashboard function
# ----------------------------
def create_engine_dashboard(
    engine_idx,
    final_predictions,
    true_rul_values,
    test_last_cycles,
    test_contradictions,
    final_mae,
    final_nasa,
    cond_trust=None,
    save=True
):
    """
    Creates a 6-panel maintenance decision dashboard for one engine.
    """

    rul_pred = final_predictions[engine_idx]
    true_rul = true_rul_values[engine_idx]
    risk_score = rul_to_risk(rul_pred)

    contradiction_flag = test_contradictions[engine_idx]
    op_cond = test_last_cycles.iloc[engine_idx]["op_cond"]

    # ----------------------------
    # Figure
    # ----------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"MAINTENANCE DECISION DASHBOARD – Engine {engine_idx + 1}",
        fontsize=16,
        fontweight="bold"
    )

    # ----------------------------
    # Panel 1: Risk
    # ----------------------------
    ax1 = plt.subplot(2, 3, 1)
    color = "green" if risk_score < 0.3 else "orange" if risk_score < 0.7 else "red"

    ax1.text(0.5, 0.6, f"RISK SCORE\n{risk_score:.2f}",
             ha="center", va="center", fontsize=20, color=color, fontweight="bold")
    ax1.text(0.5, 0.3, f"Predicted RUL: {rul_pred:.0f} cycles",
             ha="center", va="center", fontsize=13)
    ax1.axis("off")
    ax1.set_title("Current Risk Status")

    # ----------------------------
    # Panel 2: Trust Weights
    # ----------------------------
    ax2 = plt.subplot(2, 3, 2)

    if cond_trust is not None:
        try:
            w = cond_trust.loc[
                tuple(map(float, op_cond.split("_")))
            ]
            weights = [w["tabular_weight"], w["sensor_weight"], w["visual_weight"]]
        except:
            weights = [0.6, 0.3, 0.1]
    else:
        weights = [0.6, 0.3, 0.1]

    ax2.pie(
        weights,
        labels=["Tabular", "Sensor", "Visual"],
        autopct="%1.1f%%",
        colors=["#2ca02c", "#1f77b4", "#ff7f0e"]
    )
    ax2.set_title("Modality Trust Weights")

    # ----------------------------
    # Panel 3: Contradiction
    # ----------------------------
    ax3 = plt.subplot(2, 3, 3)
    if contradiction_flag:
        ax3.text(0.5, 0.5, "⚠️ CONTRADICTION\nREVIEW REQUIRED",
                 ha="center", va="center", fontsize=14, color="red", fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "✅ CONSISTENT\nPREDICTION",
                 ha="center", va="center", fontsize=14, color="green", fontweight="bold")
    ax3.axis("off")
    ax3.set_title("Data Consistency")

    # ----------------------------
    # Panel 4: Failure Indicators
    # ----------------------------
    ax4 = plt.subplot(2, 3, 4)
    features = ["sensor_15", "sensor_13", "sensor_11", "sensor_4", "sensor_9"]
    importance = [26, 18, 14, 10, 5]

    ax4.barh(features, importance, color="#1f77b4")
    ax4.set_xlabel("Relative Importance")
    ax4.set_title("Top Failure Indicators")

    # ----------------------------
    # Panel 5: Operating Context
    # ----------------------------
    ax5 = plt.subplot(2, 3, 5)
    ax5.text(
        0.5, 0.5,
        f"Operating Condition\n{op_cond}",
        ha="center", va="center", fontsize=12
    )
    ax5.axis("off")
    ax5.set_title("Operating Context")

    # ----------------------------
    # Panel 6: Summary
    # ----------------------------
    ax6 = plt.subplot(2, 3, 6)
    summary = (
        f"Global MAE: {final_mae:.2f} cycles\n"
        f"NASA Score: {final_nasa:,.0f}\n"
        f"Contradiction Rate: {test_contradictions.mean() * 100:.1f}%"
    )
    ax6.text(0.5, 0.5, summary,
             ha="center", va="center", fontsize=12, fontweight="bold")
    ax6.axis("off")
    ax6.set_title("System Performance")

    plt.tight_layout()

    if save:
        plt.savefig(f"engine_{engine_idx+1}_dashboard.png", dpi=150)

    plt.show()
