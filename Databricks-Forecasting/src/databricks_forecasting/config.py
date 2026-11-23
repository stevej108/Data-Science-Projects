REGISTERED_MODEL_NAME = (
    "workspace.default.marketing_traffic_forecast"
)

MODEL_ALIAS = "Champion"

SERVING_ENDPOINT_NAME = (
    "marketing-traffic-forecast"
)

SOURCE_TABLE = (
    "workspace.default.synthetic_model_panel_v2"
)

FORECAST_TABLE = (
    "workspace.default.future_traffic_forecast"
)

NATIONAL_FORECAST_TABLE = (
    "workspace.default.future_national_forecast"
)

FORECAST_HORIZON = 6

TRANSACTION_ASSUMPTION = (
    "latest_observation"
)


MODEL_FEATURES = [
    "Traffic_Lag_1",
    "Rolling_Traffic_3",

    "Transactions_Lag_1",
    "Transactions_Lag_3",
    "Transactions_Lag_6",

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

    "Pct_A_Facility",

    "Year",
    "Month_Number",
    "Quarter",
    "Holiday_Season",
]


STATIC_FEATURES = [
    "DMA",
    "DMA_Code",
    "Store_Count",

    "Total_Selling_SqFt",
    "Avg_Selling_SqFt",

    "DMA_Population",
    "TradeAreaPopulation",

    "Stores_per_Million",

    "CX_Composite",
    "Comp_Sales_Index",

    "Avg_Competitors",
    "Competitor_Density",

    "Pct_A_Facility",
]


TARGET = "Traffic"

VALIDATION_CUTOFF = "2026-04-01"

CANDIDATE_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "min_child_weight": 5,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

MLFLOW_EXPERIMENT_NAME = (
    "/Shared/marketing_forecasting"
)

