# ============================================================
# CONTRADICTION DETECTION MODULE
# ============================================================
# Detects disagreement between modality predictions
# Used for uncertainty-aware decision support
# ============================================================

import numpy as np

def detect_contradictions(modality_preds, threshold=30):
    """
    Detect contradictions between modalities.

    Parameters
    ----------
    modality_preds : dict
        Dictionary of modality predictions.
        Example:
        {
            "sensor": np.array([...]),
            "visual": np.array([...]),
            "tabular": np.array([...])
        }

    threshold : float
        Standard deviation threshold (cycles) above which
        modalities are considered contradictory.

    Returns
    -------
    contradictions : np.ndarray (bool)
        True if disagreement > threshold for that sample

    disagreement : np.ndarray (float)
        Standard deviation across modality predictions
    """

    # Shape → (num_modalities, num_samples)
    preds_array = np.vstack(list(modality_preds.values()))

    # Disagreement across modalities
    disagreement = np.std(preds_array, axis=0)

    # Contradiction flag
    contradictions = disagreement > threshold

    return contradictions, disagreement
