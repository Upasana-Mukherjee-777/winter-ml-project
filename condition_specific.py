def run_condition_specific_evaluation(
    train_df,
    test_df,
    rul_df,
    useful_sensors,
    MODALITIES,
    add_temporal_features,
    compute_temporal_reliability_simple,
    pca,
    visual_proxy_features,
    nasa_score
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        mean_absolute_error, accuracy_score,
        precision_score, recall_score, f1_score
    )

    # ================= TEST PREP =================
    test_df = test_df.copy()
    test_df = add_temporal_features(test_df, useful_sensors)

    pca_test = pca.transform(test_df[useful_sensors])
    for i, col in enumerate(visual_proxy_features):
        test_df[col] = pca_test[:, i]

    test_df["op_cond"] = (
        test_df[["op_setting_1", "op_setting_2"]]
        .round(1).astype(str).agg("_".join, axis=1)
    )

    # ================= TRAIN PREP =================
    train_df = train_df.copy()
    train_df = add_temporal_features(train_df, useful_sensors)

    pca_train = pca.transform(train_df[useful_sensors])
    for i, col in enumerate(visual_proxy_features):
        train_df[col] = pca_train[:, i]

    train_df["op_cond"] = (
        train_df[["op_setting_1", "op_setting_2"]]
        .round(1).astype(str).agg("_".join, axis=1)
    )

    condition_models = {}
    condition_tgns = {}

    # ================= TRAIN PER CONDITION =================
    for cond in train_df["op_cond"].unique():

        cond_df = train_df[train_df["op_cond"] == cond]
        if cond_df["engine_id"].nunique() < 3:
            continue

        engines = cond_df["engine_id"].unique()
        split = max(1, len(engines) // 5)
        val_eng, tr_eng = engines[:split], engines[split:]

        tr_df = cond_df[cond_df["engine_id"].isin(tr_eng)]
        va_df = cond_df[cond_df["engine_id"].isin(val_eng)]

        models = {}
        val_preds = {}
        valid_modalities = []

        # ---- SAFE modality training ----
        for m, feats in MODALITIES.items():
            feats = [f for f in feats if f in cond_df.columns]

            if len(feats) == 0:
                continue  # 🔴 critical safety fix

            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(tr_df[feats], tr_df["RUL"])
            models[m] = rf
            val_preds[m] = rf.predict(va_df[feats])
            valid_modalities.append(m)

        if len(valid_modalities) < 2:
            continue  # TrustGate needs ≥2 modalities

        # ---- TrustGate training ----
        rel = compute_temporal_reliability_simple(va_df, useful_sensors)
        context = va_df[["op_setting_1", "op_setting_2", "cycle_norm"]].copy()
        context["avg_reliability"] = rel.mean(axis=1)

        scaler = StandardScaler()
        X_ctx = torch.FloatTensor(scaler.fit_transform(context))
        y_true = torch.FloatTensor(va_df["RUL"].values)

        preds_tensor = {
            i: torch.FloatTensor(val_preds[m])
            for i, m in enumerate(valid_modalities)
        }

        tgn = TrustGateNetwork(X_ctx.shape[1], n_modalities=len(valid_modalities))
        opt = optim.Adam(tgn.parameters(), lr=0.001)
        loss_fn = nn.L1Loss()

        for _ in range(30):
            opt.zero_grad()
            w = tgn(X_ctx)
            fused = sum(w[:, i] * preds_tensor[i] for i in preds_tensor)
            loss = loss_fn(fused, y_true)
            loss.backward()
            opt.step()

        condition_models[cond] = (models, valid_modalities)
        condition_tgns[cond] = (tgn, scaler)

    # ================= TEST INFERENCE =================
    preds = []

    for _, df_e in test_df.groupby("engine_id"):
        sample = df_e.loc[[df_e["cycle"].idxmax()]]
        cond = sample["op_cond"].iloc[0]

        if cond not in condition_models:
            cond = list(condition_models.keys())[0]

        models, valid_modalities = condition_models[cond]
        tgn, scaler = condition_tgns[cond]

        rel = compute_temporal_reliability_simple(sample, useful_sensors)
        ctx = sample[["op_setting_1", "op_setting_2", "cycle_norm"]].copy()
        ctx["avg_reliability"] = rel.mean(axis=1)
        X_ctx = torch.FloatTensor(scaler.transform(ctx))

        preds_mod = []
        for m in valid_modalities:
            feats = [f for f in MODALITIES[m] if f in sample.columns]
            preds_mod.append(models[m].predict(sample[feats])[0])

        preds_mod = torch.FloatTensor(preds_mod)

        with torch.no_grad():
            w = tgn(X_ctx)[0]
            pred = torch.sum(w * preds_mod).item()

        preds.append(max(0.0, pred))

    preds = np.array(preds)
    true = rul_df["RUL_actual"].values

    return {
        "MAE": mean_absolute_error(true, preds),
        "NASA Score": nasa_score(true, preds),
        "Accuracy": accuracy_score(true <= 150, preds <= 150),
        "Precision": precision_score(true <= 150, preds <= 150, zero_division=0),
        "Recall": recall_score(true <= 150, preds <= 150, zero_division=0),
        "F1": f1_score(true <= 150, preds <= 150, zero_division=0),
        "Predictions": preds
    }
