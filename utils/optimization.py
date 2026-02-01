import numpy as np
from scipy.optimize import minimize


# =================================================
# Single-Channel Optimization (Safe)
# =================================================

def optimize_single_channel(model, spend_range):
    """
    Maximize predicted sales for a single spend variable.
    """

    def objective(x):
        # x arrives as array([value])
        x_val = float(x[0])                      # 🔥 SAFE scalar extraction
        X = np.array([[x_val]])                  # shape (1, 1)
        return -model.predict(X)[0]

    res = minimize(
        objective,
        x0=[np.mean(spend_range)],               # must be array-like
        bounds=[spend_range]
    )

    return {
        "optimal_spend": float(res.x[0]),
        "max_sales": float(-res.fun)
    }


# =================================================
# Single-Channel Risk-Aware Optimization (Safe)
# =================================================

def optimize_single_channel_risk_aware(
    model,
    X_train,
    y_train,
    spend_range,
    risk_lambda
):
    """
    Maximize (predicted_sales - λ * uncertainty)
    """

    # Residual-based uncertainty
    residuals = y_train - model.predict(X_train)
    sigma = np.std(residuals)

    def objective(x):
        x_val = float(x[0])                      # 🔥 SAFE scalar extraction
        X = np.array([[x_val]])                  # shape (1, 1)
        pred = model.predict(X)[0]
        return -(pred - risk_lambda * sigma)

    res = minimize(
        objective,
        x0=[np.mean(spend_range)],
        bounds=[spend_range]
    )

    return {
        "optimal_spend": float(res.x[0]),
        "risk_adjusted_sales": float(-res.fun),
        "uncertainty_sigma": float(sigma)
    }


# =================================================
# Multi-Channel Budget Optimization (Safe)
# =================================================

def optimize_budget_allocation(
    model,
    total_budget,
    bounds
):
    """
    Maximize predicted sales under a fixed total budget.
    """

    def objective(x):
        X = np.array([x], dtype=float)            # shape (1, n_features)
        return -model.predict(X)[0]

    constraints = {
        "type": "eq",
        "fun": lambda x: np.sum(x) - total_budget
    }

    x0 = np.array(
        [total_budget / len(bounds)] * len(bounds),
        dtype=float
    )

    res = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        constraints=constraints
    )

    return {
        "optimal_allocation": res.x.tolist(),
        "max_sales": float(-res.fun)
    }


# =================================================
# Multi-Channel Risk-Aware Budget Optimization (Safe)
# =================================================

def optimize_budget_allocation_risk_aware(
    model,
    X_train,
    y_train,
    total_budget,
    bounds,
    risk_lambda
):
    """
    Risk-aware multi-channel optimization:
    Maximize (predicted_sales - λ * uncertainty)
    """

    residuals = y_train - model.predict(X_train)
    sigma = np.std(residuals)

    def objective(x):
        X = np.array([x], dtype=float)            # shape (1, n_features)
        pred = model.predict(X)[0]
        return -(pred - risk_lambda * sigma)

    constraints = {
        "type": "eq",
        "fun": lambda x: np.sum(x) - total_budget
    }

    x0 = np.array(
        [total_budget / len(bounds)] * len(bounds),
        dtype=float
    )

    res = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        constraints=constraints
    )

    return {
        "optimal_allocation": res.x.tolist(),
        "risk_adjusted_sales": float(-res.fun),
        "uncertainty_sigma": float(sigma)
    }
