"""
05_forecasting.py

Purpose:
    Train and evaluate XGBoost traffic forecasting model.

Inputs:
    model_panel.parquet

Outputs:
    xgb_predictions.parquet
    feature_importance.parquet
    national_forecast.parquet
    forecast_metrics.json
"""

import json
import logging

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config import (
    PROCESSED_DATA_DIR,
    LOG_LEVEL,
)

SAVE_DATA_DIR = PROCESSED_DATA_DIR / "forecasting"

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def load_panel():

    logger.info(
        "Loading model panel..."
    )

    file_path = (
        PROCESSED_DATA_DIR /
        "model_panel.parquet"
    )

    df = pd.read_parquet(
        file_path
    )

    logger.info(
        f"Rows: {len(df):,}"
    )

    logger.info(
        f"Columns: {len(df.columns)}"
    )

    return df


# ---------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------

def get_features():

    return [

        # Marketing

        # "Spend",
        # "Spend_Lag_1",
        # "Spend_Lag_2",
        # "Spend_Lag_3",
        # "Spend_Lag_6",

        # "Rolling_Spend_3",
        # "Rolling_Spend_6",
        # "Rolling_Spend_12",

        # "Spend_Index",
        # "Spend_Share",
        # "Spend_Adstock",

        # Traffic History

        "Traffic_Lag_1",
        # "Traffic_Lag_12",

        "Rolling_Traffic_3",
        # "Traffic_YoY",

        # Transaction History

        "Transactions_Lag_1",
        "Transactions_Lag_3",
        "Transactions_Lag_6",
        # "Transactions_Lag_12",

        # "Transactions_YoY",

        # Store Attributes

        "Store_Count",
        "DMA_Code",

        "Total_Selling_SqFt",
        "Avg_Selling_SqFt",

        "DMA_Population",
        "TradeAreaPopulation",

        "Stores_per_Million",

        # CX

        "CX_Composite",

        "Comp_Sales_Index",

        "Avg_Competitors",

        "Competitor_Density",

        "Pct_A_Facility",

        # Calendar

        "Year",
        "Month_Number",

        "Quarter",

        "Holiday_Season",


        # Weather
        # "Avg_Temp",
        # "Monthly_Precip",
        # "Monthly_Snow",

        # "Hot_Weather",
        # "Cold_Weather",
        # "Heavy_Rain",
        # "Snow_Event",
        # "Heavy_Snow",
        # "Temp_Anomaly"
        

    ]


# ---------------------------------------------------------------------
# Modeling Dataset
# ---------------------------------------------------------------------

def create_modeling_dataset(df):

    df = df.copy()

    df["DMA_Code"] = (
        df["DMA"]
        .astype("category")
        .cat.codes
        )

    features = get_features()

    required = features + [
        "Traffic",
        "DMA",
        "Month"
    ]

    model_df = (
        df[required]
        .copy()
    )

    model_df["DMA_Code"] = (
        model_df["DMA"]
        .astype("category")
        .cat.codes
    )

    before_rows = len(model_df)

    model_df = model_df.dropna()

    after_rows = len(model_df)

    logger.info(
        f"Rows before dropna: "
        f"{before_rows:,}"
    )

    logger.info(
        f"Rows after dropna: "
        f"{after_rows:,}"
    )

    logger.info(
        f"Rows removed: "
        f"{before_rows - after_rows:,}"
    )

    return model_df


# ---------------------------------------------------------------------
# Train Validation Split
# ---------------------------------------------------------------------

def create_train_valid_split(df):

    train = df[
        df["Month"] <
        pd.Timestamp("2026-04-01")
    ]

    valid = df[
        df["Month"] >=
        pd.Timestamp("2026-04-01")
    ]

    logger.info(
        f"Train Rows: {len(train):,}"
    )

    logger.info(
        f"Validation Rows: {len(valid):,}"
    )

    logger.info("")
    logger.info("TRAIN PERIOD")

    logger.info(
        f"{train['Month'].min()} -> "
        f"{train['Month'].max()}"
    )

    logger.info("VALIDATION PERIOD")

    logger.info(
        f"{valid['Month'].min()} -> "
        f"{valid['Month'].max()}"
    )

    return train, valid


# ---------------------------------------------------------------------
# Tune XGBoost
# ---------------------------------------------------------------------

def tune_xgboost(
    train,
    features
):

    logger.info("")
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    X_train = train[features]

    y_train = train["Traffic"]

    param_grid = {

        "n_estimators": [
            100,
            200,
            300,
            500
        ],

        "max_depth": [
            2,
            3,
            4,
            5,
            6
        ],

        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.10
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "colsample_bytree": [
            0.6,
            0.8,
            1.0
        ],

        "min_child_weight": [
            1,
            3,
            5
        ]

    }

    model = XGBRegressor(

        objective="reg:squarederror",

        random_state=42
    )

    tscv = TimeSeriesSplit(
        n_splits=3
    )

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_grid,

        n_iter=30,

        scoring="neg_mean_absolute_error",

        cv=tscv,

        random_state=42,

        n_jobs=-1,

        verbose=1,

            )

    search.fit(
        X_train,
        y_train
    )

    best_params = search.best_params_

    with open(
            SAVE_DATA_DIR /
            "best_params.json",
            "w"
        ) as f:
    
            json.dump(
                best_params,
                f,
                indent=4
            )

    

    logger.info(
        f"Best Score: "
        f"{search.best_score_:.4f}"
    )

    logger.info(
        f"Best Params: "
        f"{search.best_params_}"
    )

    return search.best_estimator_


# ---------------------------------------------------------------------
# Train Model
# ---------------------------------------------------------------------

def train_xgboost(
    train,
    features
):

    X_train = train[features]

    y_train = train["Traffic"]

    model = XGBRegressor(

        n_estimators=500,

        max_depth=3,

        min_child_weight=5,

        learning_rate=0.03,

        subsample=0.7,

        colsample_bytree=1.0,

        random_state=42
    )

    logger.info(
        "Training XGBoost..."
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ---------------------------------------------------------------------
# Naive Benchmark
# ---------------------------------------------------------------------

def evaluate_naive_benchmark(
    valid
):

    logger.info("")
    logger.info("=" * 60)
    logger.info("NAIVE BENCHMARK")
    logger.info("=" * 60)

    actual = valid["Traffic"]

    forecast = valid["Traffic_Lag_1"]

    metrics = calculate_metrics(
        actual,
        forecast
    )

    for k, v in metrics.items():

        logger.info(
            f"{k:<15} {v:.4f}"
        )

    logger.info("=" * 60)

    return metrics

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    actual,
    forecast
):

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            forecast
        )
    )

    mae = mean_absolute_error(
        actual,
        forecast
    )

    mape = mean_absolute_percentage_error(
        actual,
        forecast
    )

    wape = (
        np.abs(
            actual - forecast
        ).sum()
        /
        np.abs(actual).sum()
    )

    r2 = r2_score(
        actual,
        forecast
    )

    metrics = {

        "RMSE": float(rmse),

        "MAE": float(mae),

        "MAPE": float(mape),

        "WAPE": float(wape),

        "R2": float(r2)
    }

    return metrics

# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------

def score_model(
    model,
    valid,
    features
):

    preds = model.predict(
        valid[features]
    )

    predictions = valid[
        [
            "DMA",
            "Month"
        ]
    ].copy()

    predictions["Actual"] = (
        valid["Traffic"]
    )

    predictions["Forecast"] = preds

    predictions["Error"] = (
        predictions["Forecast"]
        -
        predictions["Actual"]
    )

    predictions["APE"] = (
        np.abs(
            predictions["Error"]
        )
        /
        predictions["Actual"]
    )

    predictions["Pct_Error"] = (
        predictions["Forecast"]
        - predictions["Actual"]
    ) / predictions["Actual"]

    metrics = calculate_metrics(
        predictions["Actual"],
        predictions["Forecast"]
    )

    dma_performance = (
        predictions
        .groupby("DMA")
        .agg(
            Actual=("Actual","sum"),
            Forecast=("Forecast","sum"),
            APE=("APE","mean")
        )
        .sort_values(
            "APE",
            ascending=False
        )
    )

    logger.info("")
    logger.info("WORST DMA PERFORMANCE")

    logger.info(
        dma_performance
        .head(10)
        .to_string()
    )

    return predictions, metrics, dma_performance


# ---------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------

def get_feature_importance(
    model,
    features
):

    importance = pd.DataFrame({

        "Feature":
            features,

        "Importance":
            model.feature_importances_

    })

    importance = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    return importance


# ---------------------------------------------------------------------
# National Forecast
# ---------------------------------------------------------------------

def create_national_forecast(
    predictions
):

    national = (
        predictions
        .groupby(
            "Month",
            as_index=False
        )
        .agg(
            Actual=("Actual", "sum"),
            Forecast=("Forecast", "sum")
        )
    )

    national["Error"] = (
        national["Forecast"]
        -
        national["Actual"]
    )

    national_metrics = calculate_metrics(
        national["Actual"],
        national["Forecast"]
    )

    return national, national_metrics


# ---------------------------------------------------------------------
# Future Dataset Creation
# ---------------------------------------------------------------------

def create_future_periods(
    panel,
    horizon=6
):

    logger.info(
        f"Creating {horizon}-month forecast horizon..."
    )

    latest_month = (
        panel["Month"]
        .max()
    )

    future_months = pd.date_range(
        start=latest_month +
        pd.DateOffset(months=1),

        periods=horizon,

        freq="MS"
    )

    dmas = (
        panel["DMA"]
        .unique()
    )

    future = pd.MultiIndex.from_product(

        [
            dmas,
            future_months
        ],

        names=[
            "DMA",
            "Month"
        ]

    ).to_frame(
        index=False
    )

    logger.info(
        f"Future rows: {len(future):,}"
    )

    return future


# ---------------------------------------------------------------------
# Populate Static Features
# ---------------------------------------------------------------------

def populate_static_features(
    future,
    panel
):

    panel = panel.copy()

    panel["DMA_Code"] = (
        panel["DMA"]
        .astype("category")
        .cat.codes
    )
    

    latest_attributes = (

        panel

        .sort_values("Month")

        .groupby("DMA")

        .last()

        .reset_index()

    )

    cols = [

        "DMA",

        "Store_Count",
        "DMA_Code",

        "Total_Selling_SqFt",
        "Avg_Selling_SqFt",

        "DMA_Population",
        "TradeAreaPopulation",

        "Stores_per_Million",

        "CX_Composite",
        "Comp_Sales_Index",

        "Avg_Competitors",
        "Competitor_Density",

        "Pct_A_Facility"

    ]

    future = future.merge(

        latest_attributes[cols],

        on="DMA",

        how="left"

    )

    return future


# ---------------------------------------------------------------------
# Calendar Features
# ---------------------------------------------------------------------

def add_calendar_features(
    future
):

    future["Year"] = (
        future["Month"]
        .dt.year
    )

    future["Month_Number"] = (
        future["Month"]
        .dt.month
    )

    future["Quarter"] = (
        future["Month"]
        .dt.quarter
    )

    future["Holiday_Season"] = (
        future["Month_Number"]
        .isin([11, 12])
        .astype(int)
    )

    return future


# ---------------------------------------------------------------------
# Future Forecast
# ---------------------------------------------------------------------

def generate_future_forecast(
    model,
    panel,
    features,
    horizon=6
):

    logger.info(
        "Generating future forecasts..."
    )

    forecast_rows = []

    working = (
        panel
        .copy()
        .sort_values(
            ["DMA", "Month"]
        )
    )

    future = create_future_periods(
        panel,
        horizon
    )

    future = populate_static_features(
        future,
        panel
    )

    future = add_calendar_features(
        future
    )

    for month in sorted(
        future["Month"].unique()
    ):

        month_df = (
            future[
                future["Month"] == month
            ]
            .copy()
        )

        predictions = []

        for dma in month_df["DMA"]:

            history = (

                working[
                    working["DMA"] == dma
                ]

                .sort_values("Month")

            )

            latest = history.iloc[-1]

            row = (
                month_df[
                    month_df["DMA"] == dma
                ]
                .copy()
            )

            row["Traffic_Lag_1"] = (
                latest["Traffic"]
            )

            row["Rolling_Traffic_3"] = (
                history
                ["Traffic"]
                .tail(3)
                .mean()
            )

            row["Transactions_Lag_1"] = (
                latest[
                    "Transactions"
                ]
            )

            row["Transactions_Lag_3"] = (
                history[
                    "Transactions"
                ]
                .tail(3)
                .mean()
            )

            row["Transactions_Lag_6"] = (
                history[
                    "Transactions"
                ]
                .tail(6)
                .mean()
            )

            pred = model.predict(
                row[features]
            )[0]

            row["Forecast"] = pred

            predictions.append(
                row
            )

            append_row = row.copy()

            append_row["Traffic"] = pred

            append_row["Transactions"] = (
                latest[
                    "Transactions"
                ]
            )

            working = pd.concat(

                [
                    working,
                    append_row
                ],

                ignore_index=True

            )

        month_predictions = pd.concat(
            predictions
        )

        forecast_rows.append(
            month_predictions
        )

    forecast = pd.concat(
        forecast_rows
    )

    logger.info(
        f"Future forecast rows: "
        f"{len(forecast):,}"
    )

    return forecast


# ---------------------------------------------------------------------
# Future National Forecast
# ---------------------------------------------------------------------

def create_future_national_forecast(
    future_forecast
):

    logger.info(
        "Creating future national forecast..."
    )

    future_national = (

        future_forecast

        .groupby(
            "Month",
            as_index=False
        )

        .agg(
            Forecast=("Forecast", "sum")
        )

        .sort_values(
            "Month"
        )

    )

    return future_national


# ---------------------------------------------------------------------
# Future YoY Forecast
# ---------------------------------------------------------------------

def create_future_yoy_forecast(
    future_forecast,
    panel
):

    logger.info(
        "Creating future YoY forecast..."
    )

    yoy = (
        future_forecast
        .copy()
    )

    yoy["Prior_Year_Month"] = (

        yoy["Month"]
        -
        pd.DateOffset(years=1)

    )

    historical = (

        panel

        [
            [
                "DMA",
                "Month",
                "Traffic"
            ]
        ]

        .rename(
            columns={
                "Month":
                    "Prior_Year_Month",

                "Traffic":
                    "Prior_Year_Traffic"
            }
        )

    )

    yoy = yoy.merge(

        historical,

        on=[
            "DMA",
            "Prior_Year_Month"
        ],

        how="left"

    )

    yoy["YoY_Growth_Pct"] = (

        (
            yoy["Forecast"]
            -
            yoy["Prior_Year_Traffic"]
        )

        /

        yoy["Prior_Year_Traffic"]

    ) * 100

    logger.info(
        "YoY forecast rows: %s",
        len(yoy)
    )

    return yoy


# ---------------------------------------------------------------------
# National YoY Forecast
# ---------------------------------------------------------------------

def create_future_national_yoy_forecast(
    future_yoy
):

    national = (

        future_yoy

        .groupby(
            "Month",
            as_index=False
        )

        .agg(
            Forecast=(
                "Forecast",
                "sum"
            ),

            Prior_Year_Traffic=(
                "Prior_Year_Traffic",
                "sum"
            )
        )

    )

    national["YoY_Growth_Pct"] = (

        (
            national["Forecast"]
            -
            national["Prior_Year_Traffic"]
        )

        /

        national["Prior_Year_Traffic"]

    ) * 100

    return national

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_outputs(
    predictions,
    importance,
    national,
    dma_performance,
    metrics,
    national_metrics,
    naive_metrics,
    future_forecast,
    future_national,
    future_yoy,
    future_national_yoy
):

    predictions.to_parquet(
        SAVE_DATA_DIR /
        "xgb_predictions.parquet",
        index=False
    )

    importance.to_parquet(
        SAVE_DATA_DIR /
        "feature_importance.parquet",
        index=False
    )

    national.to_parquet(
        SAVE_DATA_DIR /
        "national_forecast.parquet",
        index=False
    )

    dma_performance.to_parquet(
        SAVE_DATA_DIR /
        "dma_performance.parquet"
    )

    future_forecast.to_parquet(

        PROCESSED_DATA_DIR /
        "forecasting" /
        "future_forecast.parquet",

        index=False

    )

    future_national.to_parquet(

        PROCESSED_DATA_DIR /
        "forecasting" /
        "future_national_forecast.parquet",

        index=False

    )

    future_yoy.to_parquet(

        PROCESSED_DATA_DIR /
        "forecasting" /
        "future_yoy_forecast.parquet",

        index=False

    )

    future_national_yoy.to_parquet(

        PROCESSED_DATA_DIR /
        "forecasting" /
        "future_national_yoy_forecast.parquet",

        index=False

    )

    all_metrics = {

        "naive_metrics":
            naive_metrics,

        "dma_metrics":
            metrics,

        "national_metrics":
            national_metrics
    }

    wape_improvement = (
        naive_metrics["WAPE"]
        -
        metrics["WAPE"]
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("XGBOOST vs NAIVE")
    logger.info("=" * 60)

    logger.info(
        f"WAPE Improvement: "
        f"{wape_improvement:.4f}"
    )

    logger.info(
        f"Relative Improvement: "
        f"{wape_improvement / naive_metrics['WAPE']:.1%}"
    )

    logger.info("=" * 60)

    summary = pd.DataFrame({

        "Metric": [
            "DMA_WAPE",
            "DMA_MAPE",
            "DMA_R2",
            "National_WAPE",
            "National_MAPE"
        ],

        "Value": [
            metrics["WAPE"],
            metrics["MAPE"],
            metrics["R2"],
            national_metrics["WAPE"],
            national_metrics["MAPE"]
        ]
    })

    summary.to_parquet(
            SAVE_DATA_DIR /
            "validation_summary.parquet"
        )


    with open(
        SAVE_DATA_DIR /
        "forecast_metrics.json",
        "w"
    ) as f:

        json.dump(
            all_metrics,
            f,
            indent=4
        )

    logger.info(
        "Forecast outputs saved."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    panel = load_panel()

    features = get_features()

    modeling_df = (
        create_modeling_dataset(
            panel
        )
    )

    train, valid = (
        create_train_valid_split(
            modeling_df
        )
    )

    naive_metrics = evaluate_naive_benchmark(
        valid
    )

    model = train_xgboost(
        train,
        features
    )

    predictions, metrics, dma_performance =(
        score_model(
            model,
            valid,
            features
        )
    )

    importance = (
        get_feature_importance(
            model,
            features
        )
    )

    logger.info("\nTOP FEATURES")

    logger.info(
        importance.head(20).to_string(index=False)
    )

    logger.info("\nLOWEST FEATURES")

    logger.info(
        importance.tail(15).to_string(index=False)
    )


    national, national_metrics = (
        create_national_forecast(
            predictions
        )
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("NATIONAL PERFORMANCE")
    logger.info("=" * 60)

    for k, v in national_metrics.items():

        logger.info(
            f"{k:<15} {v:.4f}"
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 60)

    for k, v in metrics.items():

        logger.info(
            f"{k:<15} {v:.4f}"
        )

    logger.info("=" * 60)


    future_forecast = (
            generate_future_forecast(
                model,
                panel,
                features,
                horizon=6
            )
    )

    future_national = (
        create_future_national_forecast(
            future_forecast
        )
    )

    future_yoy = (
        create_future_yoy_forecast(
            future_forecast,
            panel
        )
    )

    future_national_yoy = (
        create_future_national_yoy_forecast(
            future_yoy
        )
    )
        

    logger.info("")
    logger.info("=" * 60)
    logger.info("FUTURE FORECAST SUMMARY")
    logger.info("=" * 60)

    logger.info(
        future_forecast.groupby("Month")
        ["Forecast"]
        .sum()
        .to_string()
    )

    logger.info("")

    logger.info(
        future_forecast["Forecast"]
        .describe()
        .to_string()
    )
   

    save_outputs(
        predictions,
        importance,
        national,
        dma_performance,
        metrics,
        national_metrics,
        naive_metrics,
        future_forecast,
        future_national,
        future_yoy,
        future_national_yoy
    )


if __name__ == "__main__":

    main()

