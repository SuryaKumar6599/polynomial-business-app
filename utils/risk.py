import numpy as np


def extrapolation_risk(user_input, train_data):
    flags = []
    for i in range(train_data.shape[1]):
        if user_input[i] < train_data[:, i].min() or user_input[i] > train_data[:, i].max():
            flags.append(i)

    return {
        "is_risky": len(flags) > 0,
        "feature_indices": flags
    }


def diminishing_returns(spend, sales, threshold=0.05):
    marginal = np.diff(sales) / np.diff(spend)
    idx = np.where(marginal < threshold)[0]

    return {
        "has_diminishing_returns": len(idx) > 0,
        "saturation_index": int(idx[0]) if len(idx) > 0 else None
    }


def retraining_required(anomaly_count, threshold=5):
    return anomaly_count >= threshold
