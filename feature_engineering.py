import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# TEMPORAL FEATURES
# ---------------------------------------------------------
def add_temporal_features(df, sensors, roll_window=10, slope_window=20):
    df = df.copy()

    for s in sensors:
        df[f"{s}_roll_mean"] = (
            df.groupby("engine_id")[s]
            .rolling(roll_window, min_periods=1)
            .mean()
            .values
        )

        df[f"{s}_roll_std"] = (
            df.groupby("engine_id")[s]
            .rolling(roll_window, min_periods=1)
            .std()
            .fillna(0)
            .values
        )

        def slope(x):
            return np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0

        df[f"{s}_slope"] = (
            df.groupby("engine_id")[s]
            .apply(lambda g: g.rolling(slope_window, min_periods=2).apply(slope))
            .fillna(0)
            .values
        )

    df["cycle_norm"] = (
        df["cycle"] /
        df.groupby("engine_id")["cycle"].transform("max")
    )

    return df


# ---------------------------------------------------------
# PCA VISUAL PROXY (VARIANCE-BASED)
# ---------------------------------------------------------
def add_visual_proxy_features(df, sensors, variance_threshold=0.90):
    df = df.copy()

    pca = PCA(n_components=variance_threshold, random_state=42)
    visual_data = pca.fit_transform(df[sensors])

    visual_features = [
        f"visual_proxy_{i+1}" for i in range(pca.n_components_)
    ]

    for i, col in enumerate(visual_features):
        df[col] = visual_data[:, i]

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": pca.explained_variance_ratio_.cumsum(),
        "n_components": pca.n_components_
    }

    return df, visual_features, meta


# ---------------------------------------------------------
# MODALITY DEFINITIONS
# ---------------------------------------------------------
def define_modalities(useful_sensors, visual_features):
    sensor_features = []
    for s in useful_sensors:
        sensor_features.extend([
            s,
            f"{s}_roll_mean",
            f"{s}_roll_std",
            f"{s}_slope"
        ])

    tabular_features = ["op_setting_1", "op_setting_2", "cycle_norm"]

    modalities = {
        "sensor": sensor_features,
        "visual": visual_features,
        "tabular": tabular_features
    }

    all_features = sensor_features + visual_features + tabular_features

    return modalities, all_features
