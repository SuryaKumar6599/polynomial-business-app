import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.polynomial import (
    PolynomialRegressionModel,
    RidgePolynomialRegressionModel,
    LinearRegressionModel
)

from utils.metrics import (
    regression_metrics,
    prediction_confidence_band
)

from utils.risk import (
    extrapolation_risk,
    diminishing_returns,
    retraining_required
)

from utils.optimization import (
    optimize_single_channel,
    optimize_single_channel_risk_aware,
    optimize_budget_allocation,
    optimize_budget_allocation_risk_aware
)

from utils.logging import logger

# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(
    page_title="Polynomial Regression – Business Analytics",
    layout="wide"
)

st.title("📊 Polynomial Regression – Business Forecasting & Optimization")
st.caption("Model Version: v1.0.0")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/advertising.csv")

df = load_data()

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tabs = st.tabs([
    "Overview",
    "Single-Variable Analysis",
    "Multi-Variable Analysis",
    "Model Comparison",
    "Forecasting",
    "Optimization",
    "Risk & Diagnostics"
])

# =================================================
# TAB 1 — Overview
# =================================================
with tabs[0]:
    st.header("Business Context")
    st.markdown("""
    This application demonstrates **end-to-end applied machine learning**
    using polynomial regression for **business forecasting and optimization**.

    **Key capabilities**
    - Polynomial & Ridge regression
    - Auto degree selection (cross-validation)
    - Confidence intervals & uncertainty
    - Risk-aware optimization
    - Model comparison
    - Drift-based anomaly detection
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# =================================================
# TAB 2 — Single-Variable Polynomial Regression
# =================================================
with tabs[1]:
    st.header("Single-Variable Polynomial Regression (TV → Sales)")

    mode = st.radio(
        "Degree Selection Mode",
        ["Manual", "Auto (Cross-Validation)"],
        horizontal=True
    )

    X = df[["TV"]].values
    y = df[TARGET].values

    if mode == "Manual":
        degree = st.slider("Polynomial Degree", 1, 5, 2)
        model = PolynomialRegressionModel(degree=degree).fit(X, y)

    else:
        cv_result = PolynomialRegressionModel.auto_select_degree(
            X, y, max_degree=5, cv=5
        )
        degree = cv_result["best_degree"]
        model = PolynomialRegressionModel(degree=degree).fit(X, y)

        st.success(f"Auto-selected Degree: {degree}")

        fig, ax = plt.subplots()
        ax.plot(
            list(cv_result["cv_scores"].keys()),
            list(cv_result["cv_scores"].values()),
            marker="o"
        )
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("CV R² Score")
        st.pyplot(fig)

    # Confidence bands
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    conf = prediction_confidence_band(model, X, y, X_range)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.4, label="Actual")
    ax.plot(X_range, conf["mean"], label="Prediction")

    ax.fill_between(
        X_range.flatten(),
        conf["lower_1sigma"],
        conf["upper_1sigma"],
        alpha=0.3,
        label="±1σ"
    )

    ax.fill_between(
        X_range.flatten(),
        conf["lower_2sigma"],
        conf["upper_2sigma"],
        alpha=0.15,
        label="±2σ"
    )

    ax.set_xlabel("TV Spend")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    if conf["sigma"] > 5:
        st.warning("⚠️ High uncertainty detected in predictions.")

    st.subheader("Model Metrics")
    st.json(model.evaluate(X, y))

# =================================================
# TAB 3 — Multi-Variable Regression
# =================================================
with tabs[2]:
    st.header("Multi-Variable Polynomial Regression")

    degree_mv = st.slider("Polynomial Degree", 1, 4, 2)

    model_type = st.radio(
        "Model Type",
        ["Polynomial", "Ridge Polynomial"],
        horizontal=True
    )

    alpha = None
    if model_type == "Ridge Polynomial":
        alpha = st.slider("Ridge α", 0.01, 10.0, 1.0)

    X_mv = df[FEATURES].values
    y = df[TARGET].values

    if model_type == "Polynomial":
        model_mv = PolynomialRegressionModel(degree_mv).fit(X_mv, y)
    else:
        model_mv = RidgePolynomialRegressionModel(degree_mv, alpha).fit(X_mv, y)

    preds = model_mv.predict(X_mv)

    st.subheader("Performance")
    st.json(regression_metrics(y, preds))

    logger.info(f"Multi-variable model trained: {model_type}, degree={degree_mv}")

# =================================================
# TAB 4 — Model Comparison
# =================================================
with tabs[3]:
    st.header("Model Comparison Dashboard")

    degree = st.slider("Polynomial Degree (Comparison)", 1, 4, 2)
    alpha = st.slider("Ridge α (Comparison)", 0.01, 10.0, 1.0)

    X = df[FEATURES].values
    y = df[TARGET].values

    models = {
        "Linear": LinearRegressionModel().fit(X, y),
        "Polynomial": PolynomialRegressionModel(degree).fit(X, y),
        "Ridge Polynomial": RidgePolynomialRegressionModel(degree, alpha).fit(X, y)
    }

    results = []
    for name, m in models.items():
        preds = m.predict(X)
        metrics = regression_metrics(y, preds)
        results.append({
            "Model": name,
            "R²": metrics["r2"],
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"]
        })

    st.dataframe(pd.DataFrame(results))

# =================================================
# TAB 5 — Forecasting
# =================================================
with tabs[4]:
    st.header("Sales Forecasting (What-If Analysis)")

    tv = st.slider("TV Spend", 0.0, float(df.TV.max() * 1.5), float(df.TV.mean()))
    radio = st.slider("Radio Spend", 0.0, float(df.Radio.max() * 1.5), float(df.Radio.mean()))
    news = st.slider("Newspaper Spend", 0.0, float(df.Newspaper.max() * 1.5), float(df.Newspaper.mean()))

    input_data = np.array([[tv, radio, news]])

    forecast = model_mv.predict(input_data)[0]
    st.metric("Predicted Sales", f"{forecast:.2f}")

    risk = extrapolation_risk(input_data.flatten(), df[FEATURES].values)
    if risk["is_risky"]:
        st.warning("⚠️ Forecast uses extrapolated spend values.")

    logger.info(f"Forecast requested: {input_data.tolist()}")

# =================================================
# TAB 6 — Optimization
# =================================================
with tabs[5]:
    st.header("Advertising Spend Optimization")

    # Single-channel
    st.subheader("Risk-Aware Single-Channel Optimization (TV)")

    risk_lambda = st.slider("Risk Aversion (λ)", 0.0, 5.0, 1.0, 0.1)

    single_model = PolynomialRegressionModel(2).fit(df[["TV"]].values, y)

    res = optimize_single_channel_risk_aware(
        single_model,
        df[["TV"]].values,
        y,
        (df.TV.min(), df.TV.max()),
        risk_lambda
    )

    st.success(
        f"Optimal TV Spend: {res['optimal_spend']:.2f} | "
        f"Risk-Adjusted Sales: {res['risk_adjusted_sales']:.2f}"
    )

    # Multi-channel
    st.subheader("Risk-Aware Multi-Channel Optimization")

    total_budget = st.slider("Total Budget", 50.0, 500.0, 200.0)
    bounds = [(0, total_budget)] * 3

    res_mc = optimize_budget_allocation_risk_aware(
        model_mv,
        df[FEATURES].values,
        y,
        total_budget,
        bounds,
        risk_lambda
    )

    st.json(dict(zip(FEATURES, res_mc["optimal_allocation"])))
    st.metric("Risk-Adjusted Max Sales", f"{res_mc['risk_adjusted_sales']:.2f}")

# =================================================
# TAB 7 — Risk & Diagnostics
# =================================================
with tabs[6]:
    st.header("Risk & Diagnostics")

    # Diminishing returns
    spend_range = np.linspace(df.TV.min(), df.TV.max(), 200)
    sales_curve = single_model.predict(spend_range.reshape(-1, 1))

    risk_info = diminishing_returns(spend_range, sales_curve)

    if risk_info["has_diminishing_returns"]:
        st.warning(
            f"Diminishing returns beyond TV spend ≈ "
            f"{spend_range[risk_info['saturation_index']]:.2f}"
        )

    # Residual drift
    st.subheader("Residual Drift (Synthetic Time)")

    df_time = df.copy()
    df_time["time"] = np.arange(len(df_time))

    preds = model_mv.predict(df_time[FEATURES].values)
    residuals = df_time[TARGET] - preds

    window = st.slider("Rolling Window", 5, 30, 10)
    roll_mean = residuals.rolling(window).mean()
    roll_std = residuals.rolling(window).std()

    upper = roll_mean + 2 * roll_std
    lower = roll_mean - 2 * roll_std

    anomalies = (residuals > upper) | (residuals < lower)
    anomaly_count = anomalies.sum()

    fig, ax = plt.subplots()
    ax.plot(residuals.values, label="Residuals")
    ax.plot(roll_mean.values, label="Rolling Mean")
    ax.fill_between(
        range(len(residuals)),
        lower.values,
        upper.values,
        alpha=0.2,
        label="±2σ"
    )
    ax.scatter(
        np.where(anomalies)[0],
        residuals[anomalies],
        color="red",
        label="Anomaly"
    )
    ax.legend()
    st.pyplot(fig)

    if retraining_required(anomaly_count):
        st.error("🚨 Model retraining recommended due to persistent drift.")
