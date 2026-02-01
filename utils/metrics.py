import numpy as np


def regression_metrics(y_true, y_pred):
    return {
        "r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2),
        "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "mae": np.mean(np.abs(y_true - y_pred))
    }


def marginal_roi(spend, sales):
    return np.diff(sales) / np.diff(spend)


def prediction_confidence_band(model, X_train, y_train, X_pred):
    train_preds = model.predict(X_train)
    sigma = np.std(y_train - train_preds)

    preds = model.predict(X_pred)
    return {
        "mean": preds,
        "lower_1sigma": preds - sigma,
        "upper_1sigma": preds + sigma,
        "lower_2sigma": preds - 2 * sigma,
        "upper_2sigma": preds + 2 * sigma,
        "sigma": sigma
    }
