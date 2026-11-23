# Databricks Forecasting MLOps Pipeline

An end-to-end forecasting and MLOps project built on Databricks that automates the machine learning lifecycle from candidate training through model validation, governance, production deployment, and six-month forward batch forecasting.

The project forecasts traffic at the DMA level using XGBoost, evaluates new candidates against explicit performance gates, manages model versions through MLflow and Unity Catalog, automatically deploys the approved `Champion` model to Databricks Model Serving, and generates recursive future forecasts persisted as Delta tables.

The objective is not simply to train an accurate forecasting model. The project demonstrates how a model can move through a repeatable, testable, and governed production lifecycle.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Repository Architecture](#repository-architecture)
- [Development Workflow](#development-workflow)
- [Data and Modeling Logic](#data-and-modeling-logic)
- [Feature Set](#feature-set)
- [Candidate Training](#candidate-training)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Candidate Validation and Model Governance](#candidate-validation-and-model-governance)
- [MLflow Lifecycle](#mlflow-lifecycle)
- [Unity Catalog Model Registry](#unity-catalog-model-registry)
- [Champion Model Pattern](#champion-model-pattern)
- [Model Serving](#model-serving)
- [Batch Forecasting](#batch-forecasting)
- [Forecast Assumptions](#forecast-assumptions)
- [Delta Lake Outputs](#delta-lake-outputs)
- [Model Lineage](#model-lineage)
- [Testing Strategy](#testing-strategy)
- [Failure Safety](#failure-safety)
- [Productionization Lessons](#productionization-lessons)
- [Configuration](#configuration)
- [Current Limitations](#current-limitations)
- [Potential Future Directions](#potential-future-directions)
- [Design Principles](#design-principles)
- [Summary](#summary)

---

# Project Overview

The system implements a complete ML forecasting workflow across five automated Databricks Job tasks:

1. Train a new candidate model
2. Validate the candidate against explicit performance gates
3. Register and promote approved candidates
4. Deploy the promoted model to Databricks Model Serving
5. Generate and persist six-month forward forecasts

The workflow connects:

```text
Data Engineering
        ↓
Feature Engineering
        ↓
Model Training
        ↓
MLflow Tracking
        ↓
Model Evaluation
        ↓
Automated Governance
        ↓
Unity Catalog
        ↓
Model Serving
        ↓
Batch Inference
        ↓
Delta Lake
```

Important model identifiers such as the candidate MLflow Run ID, promoted model version, and deployed model version are passed between tasks rather than manually copied or rediscovered.

This creates explicit lineage from the candidate training run through production inference.

---

# Technology Stack

The project uses:

- Python
- pandas
- XGBoost
- PySpark
- Databricks
- Databricks Jobs
- Databricks Model Serving
- Delta Lake
- MLflow
- Unity Catalog
- pytest
- Git
- VS Code

The architecture deliberately separates local source development from Databricks execution.

---

# System Architecture

```text
                         Historical Delta Table
                       synthetic_model_panel_v2
                                  |
                                  v
                    +---------------------------+
                    | 01 - Train Candidate      |
                    |                           |
                    | Feature engineering       |
                    | Time-based validation     |
                    | Naive benchmark           |
                    | XGBoost training          |
                    | MLflow logging            |
                    +-------------+-------------+
                                  |
                           candidate_run_id
                                  |
                                  v
                    +---------------------------+
                    | 02 - Validate Candidate   |
                    |                           |
                    | Candidate vs naive        |
                    | Maximum WAPE gate         |
                    | Improvement gate          |
                    +-------------+-------------+
                                  |
                         PASS ----+---- FAIL
                           |              |
                           v             STOP
                    +---------------------------+
                    | 03 - Promote Candidate    |
                    |                           |
                    | Register model version    |
                    | Assign Champion alias     |
                    | Verify promotion          |
                    +-------------+-------------+
                                  |
                       promoted_model_version
                                  |
                                  v
                    +---------------------------+
                    | 04 - Update Endpoint      |
                    |                           |
                    | Compare deployed version  |
                    | Deploy new Champion       |
                    | Wait for readiness        |
                    | Verify deployment         |
                    +-------------+-------------+
                                  |
                       deployed_model_version
                                  |
                                  v
                    +---------------------------+
                    | 05 - Batch Forecast       |
                    |                           |
                    | Generate future periods   |
                    | Recursive inference       |
                    | Schema enforcement        |
                    | National aggregation      |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
       future_traffic_forecast       future_national_forecast
             Delta Table                    Delta Table
```

The architecture separates each important lifecycle transition.

A candidate being trained does not imply it should be promoted.

A model being promoted does not prove thatendpoint
│   └── 05_batch_forecast
|
├── src/
│   └── databricks_forecasting/
│       ├── config.py
│       ├── deployment.py
│       ├── evaluation.py
│       ├── forecasting.py
│       ├── model_wrapper.py
│       └── training.py
|
├── tests/
│   ├── ...
│   ├── test_validate_candidate.py
│   ├── test_promote_candidate.py
│   ├── test_deployment.py
│   ├── test_update_endpoint.py
│   ├── test_forecasting.py
│   └── test_batch_forecast.py
|
├── pyproject.toml
├── requirements.txt
└── README.md
```

## `src/databricks_forecasting`

Contains reusable modeling and application logic.

Responsibilities include:

- data preparation
- feature engineering
- training
- evaluation
- MLflow model wrapping
- recursive forecasting
- serving endpoint configuration logic

This layer is designed to remain independently testable.

## `jobs`

Contains production entry points that connect the reusable application code to Databricks services.

These scripts interact with:

- Spark
- MLflow
- Unity Catalog
- Model Serving
- Delta Lake

## `orchestration`

Contains thin Databricks notebooks.

Their primary responsibilities are:

```text
Receive task parameter
        ↓
Invoke tested application code
        ↓
Publish downstream task value
```

Business logic is intentionally kept out of orchestration notebooks.

## `tests`

Contains automated tests covering the modeling, governance, registry, deployment, and forecasting layers.

External systems are mocked where appropriate so tests do not mutate production MLflow or Databricks state.

---

# Development Workflow

VS Code is treated as the primary source-development environment.

Git is the source of truth.

The workflow is:

```text
VS Code
   |
   v
Implement / Refactor
   |
   v
pytest
   |
   v
Git Commit
   |
   v
Git Push
   |
   v
Databricks Git Pull
   |
   v
Databricks Job Execution
```

Production code is therefore not independently developed inside the Databricks workspace.

This provides a reproducible relationship between:

```text
code written locally
        =
code tested locally
        =
code committed to Git
        =
code executed in Databricks
```

---

# Data and Modeling Logic

The source modeling panel is stored in:

```text
workspace.default.synthetic_model_panel_v2
```

The modeling grain is:

```text
DMA × Month
```

The prediction target is:

```text
Traffic
```

The current data is synthetic but structured to support the same types of temporal, operational, geographic, competitive, and behavioral relationships expected in a production forecasting system.

The pipeline is therefore designed around production-style architecture even though the current modeling dataset is synthetic.

---

# Feature Set

The model combines historical traffic behavior, transaction activity, operating characteristics, geography, competitive signals, customer experience measures, and calendar effects.

## Traffic history

```text
Traffic_Lag_1
Rolling_Traffic_3
```

These capture recent traffic level and short-term momentum.

## Transaction history

```text
Transactions_Lag_1
Transactions_Lag_3
Transactions_Lag_6
```

These represent short-, medium-, and longer-term transaction behavior.

## Store network characteristics

```text
Store_Count
Total_Selling_SqFt
Avg_Selling_SqFt
Stores_per_Million
```

## Market characteristics

```text
DMA_Population
TradeAreaPopulation
```

## Customer and commercial signals

```text
CX_Composite
Comp_Sales_Index
```

## Competitive environment

```text
Avg_Competitors
Competitor_Density
```

## Facility characteristics

```text
Pct_A_Facility
```

## Calendar effects

```text
Year
Month_Number
Quarter
Holiday_Season
```

The exact model feature order is centralized in configuration.

This allows training and production inference to operate against the same feature contract.

---

# Candidate Training

The first Databricks Job task produces a new candidate model.

Conceptually:

```text
Historical Panel
      ↓
Modeling Dataset
      ↓
Time-Based Train / Validation Split
      ↓
Naive Benchmark
      ↓
XGBoost Candidate
      ↓
Candidate Evaluation
      ↓
MLflow Logging
      ↓
candidate_run_id
```

The current validation cutoff is:

```text
2026-04-01
```

The candidate currently uses XGBoost.

Current hyperparameter configuration:

```python
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
```

The training stage is responsible for creating the candidate artifact.

It does **not** decide whether that artifact should become the production model.

---

# Evaluation and Benchmarking

Candidate performance is evaluated against a naive forecasting benchmark.

This is important because model performance is not evaluated in isolation.

A complex model should demonstrate measurable value over a simple forecasting strategy.

The training process records:

```text
validation_rmse
validation_mae
validation_mape
validation_wape
validation_r2

naive_wape
relative_wape_improvement
```

Relative improvement is calculated by comparing candidate WAPE with benchmark WAPE.

One representative successful candidate during development produced:

```text
Candidate WAPE:        12.29%
Naive WAPE:            15.25%
Relative improvement:  19.38%
```

Those metrics are logged to MLflow and consumed by the next pipeline stage.

---

# Candidate Validation and Model Governance

Training success does not automatically trigger promotion.

The validation task retrieves the exact MLflow candidate run produced by Task 01 and evaluates it against explicit promotion criteria.

Current requirements are:

| Validation Gate | Requirement |
|---|---:|
| Candidate performance | Candidate must outperform naive |
| Maximum validation WAPE | ≤ 15% |
| Minimum relative WAPE improvement | ≥ 5% |

A passing candidate may resemble:

```text
Candidate WAPE:        12.29%
Naive WAPE:            15.25%
Relative improvement:  19.38%

PASS: Candidate beats naive baseline
PASS: Validation WAPE <= 15%
PASS: Relative WAPE improvement >= 5%

VALIDATION RESULT: PASS
```

If any gate fails:

```text
Candidate
    ↓
Validation FAIL
    ↓
Task fails
    ↓
Promotion never executes
```

This establishes an automated governance boundary between experimentation and production.

---

# MLflow Lifecycle

MLflow plays several separate roles throughout the pipeline.

Understanding these roles is important because experiment tracking, model packaging, model registration, and deployment are related but distinct concepts.

---

## 1. Experiment Tracking

Each candidate model is trained inside an MLflow run.

The run records parameters such as:

```text
candidate hyperparameters
validation cutoff
feature count
training row count
validation row count
```

and metrics such as:

```text
RMSE
MAE
MAPE
WAPE
R2
Naive WAPE
Relative WAPE Improvement
```

Each run receives an immutable Run ID.

For example:

```text
a80b562af8be48a59d222dd9dd0d1d76
```

The exact Run ID produced during training is passed to validation through Databricks task values.

This means validation does not search for "the latest run." It validates the exact model candidate generated upstream.

---

## 2. Model Packaging

The XGBoost model is wrapped inside a custom MLflow PyFunc model:

```text
MarketingForecastPyFunc
```

The logged model artifact contains more than the trained estimator.

Conceptually:

```text
MLflow Model Artifact
|
├── trained XGBoost model
├── custom PythonModel wrapper
├── input signature
├── output signature
├── input example
├── package requirements
└── custom databricks_forecasting source code
```

This was an important productionization requirement.

Code available in the Databricks training workspace is not automatically available in the isolated Databricks Model Serving environment.

During development, serving initially failed with:

```text
ModuleNotFoundError:
No module named 'databricks_forecasting'
```

The final training process therefore packages the required custom source alongside the MLflow model.

The model artifact is consequently portable beyond the original training session.

---

## 3. Model Signature

MLflow records an explicit input/output contract for the model.

Representative input types include:

```text
Traffic_Lag_1          double
Rolling_Traffic_3      double

Transactions_Lag_1     double
Transactions_Lag_3     double
Transactions_Lag_6     double

Store_Count            long

Year                   integer
Month_Number           integer
Quarter                integer
Holiday_Season         long
```

This signature is enforced during prediction.

That behavior exposed an actual production issue during batch forecasting.

Calendar features generated through pandas were logically correct but did not always preserve the exact integer dtypes expected by the registered MLflow model.

For example:

```text
Model expects:
Year -> int32

Batch input produced:
Year -> int64
```

Instead of disabling schema enforcement, the project explicitly normalizes model inputs immediately before inference.

The final flow is:

```text
Generate Features
      ↓
Select MODEL_FEATURES
      ↓
Enforce Input Dtypes
      ↓
MLflow Signature Validation
      ↓
Model Prediction
```

The model signature therefore acts as an active production contract.

---

# Unity Catalog Model Registry

Approved models are registered in Unity Catalog as:

```text
workspace.default.marketing_traffic_forecast
```

Each successful promotion creates a new immutable model version.

Conceptually:

```text
Validated MLflow Run
       ↓
Register Model
       ↓
Create Version N
       ↓
Assign Champion -> Version N
       ↓
Read Champion Back
       ↓
Verify Version N
```

The promotion task does not merely submit the registry operation.

It reads the alias back and verifies that the expected version became `Champion`.

---

# Champion Model Pattern

The production alias is:

```text
Champion
```

Model versions answer:

> Which immutable model artifact is this?

The alias answers:

> Which model has been approved for production use?

For example:

```text
Version 8
Version 9
Version 10 ← Champion
```

Downstream logic therefore does not need to permanently encode a specific version.

The production designation can move as new models are approved.

This separates model history from model role.

---

# Model Serving

The Databricks Model Serving endpoint is:

```text
marketing-traffic-forecast
```

Promotion and deployment are intentionally separate stages.

A model becoming `Champion` does not prove that the Databricks serving environment successfully loaded it.

Task 04 therefore performs a controlled deployment transition.

```text
promoted_model_version
        ↓
Verify version == Champion
        ↓
Retrieve serving endpoint
        ↓
Determine deployed version
        ↓
Compare deployed vs target
```

---

## Idempotent Deployment

If:

```text
deployed_version == Champion
```

the pipeline performs no deployment.

```text
Endpoint already current
       ↓
No-op
       ↓
SUCCESS
```

If:

```text
deployed_version != Champion
```

the pipeline updates the endpoint:

```text
Old Deployment
      ↓
Update Configuration
      ↓
Wait for Databricks Deployment
      ↓
Retrieve Endpoint Again
      ↓
Verify New Version
```

This prevents unnecessary redeployment during repeated workflow executions.

---

## Deployment Verification

Endpoint configuration updates are asynchronous.

The pipeline therefore waits for Databricks to finish the deployment operation before continuing.

It then independently retrieves the endpoint state and confirms the deployed model version.

Task 05 does not run unless deployment completes successfully.

During final pipeline validation, the endpoint reached:

```text
Ready
```

while serving:

```text
V10
```

---

# Batch Forecasting

Once deployment has succeeded, the final task generates the future traffic forecast.

The configured horizon is:

```text
6 months
```

Forecasting occurs at the:

```text
DMA × Month
```

level.

The process is recursive because Traffic is itself part of the feature history needed for future periods.

---

## Future Period Generation

The pipeline identifies the latest historical month:

```python
latest_month = panel["Month"].max()
```

Future periods begin one month later.

For a six-month horizon:

```text
Latest Historical Month
          ↓
       Month +1
       Month +2
       Month +3
       Month +4
       Month +5
       Month +6
```

The system builds the cross product of:

```text
All DMAs
×
All Future Months
```

and verifies that the future period does not overlap historical observations.

---

## Static Feature Preparation

For each DMA, the most recent available values of relatively static characteristics are carried into the future frame.

These include:

```text
Store_Count
Total_Selling_SqFt
Avg_Selling_SqFt
DMA_Population
TradeAreaPopulation
Stores_per_Million
CX_Composite
Comp_Sales_Index
Avg_Competitors
Competitor_Density
Pct_A_Facility
```

This is a deliberate forecasting assumption.

It can later be replaced for individual variables when reliable future planning inputs become available.

---

## Calendar Features

Future months generate:

```text
Year
Month_Number
Quarter
Holiday_Season
```

These values are normalized to the precise dtypes expected by the MLflow model signature before prediction.

---

## Recursive Traffic Forecasting

The key batch forecasting logic is recursive.

For the first future month:

```text
Historical Traffic
       ↓
Traffic_Lag_1
Rolling_Traffic_3
       ↓
Model
       ↓
Forecast Month +1
```

Once Month +1 is predicted:

```text
Forecast Month +1
       ↓
becomes future Traffic history
       ↓
recalculate lag features
       ↓
Forecast Month +2
```

The process repeats:

```text
Observed History
      |
      v
Forecast H1
      |
      v
Forecast H2
      |
      v
Forecast H3
      |
      v
Forecast H4
      |
      v
Forecast H5
      |
      v
Forecast H6
```

This allows the lag-based model to operate beyond the point where actual Traffic observations exist.

---

# Forecast Assumptions

Forecasting requires assumptions about variables that are not known in advance.

Those assumptions are made explicit rather than hidden inside the implementation.

---

## Traffic

Traffic is recursively forecast.

Predicted Traffic becomes future Traffic history for subsequent horizons.

---

## Transactions

Future Transactions currently use:

```text
latest_observation
```

as the configured strategy.

The most recently observed transaction level is carried forward during recursive forecasting.

Transaction lag variables are then calculated from that extended history.

This provides a simple and reproducible assumption while keeping the architecture open to a future transaction forecast.

---

## Static DMA Characteristics

Most geographic, operational, customer, and competitive attributes currently use the most recent observed value.

The architecture could later accept known future values without restructuring the model lifecycle.

---

# Delta Lake Outputs

The pipeline produces two managed production forecast tables.

---

## DMA-Level Traffic Forecast

```text
workspace.default.future_traffic_forecast
```

The detailed table contains forecasts by DMA and future month.

Relevant fields include:

```text
DMA
Month
Forecast
Forecast_Horizon
Model_Version
```

along with the associated feature context.

---

## National Traffic Forecast

```text
workspace.default.future_national_forecast
```

The national forecast is calculated from the DMA output.

```text
DMA Forecasts
      ↓
Group by Future Month
      ↓
Sum Forecast
      ↓
National Forecast
```

This guarantees that the national result reconciles with the detailed DMA forecasts.

A separate national model is not used.

---

# Forecast Snapshot Strategy

The production forecast tables use overwrite semantics.

Every successful workflow execution creates the new authoritative forecast snapshot.

```text
Previous Forecast
       ↓
New Successful Pipeline Run
       ↓
Overwrite
       ↓
Current Forecast
```

This avoids silently generating duplicate predictions such as:

```text
DMA_A | 2027-01 | 500000
DMA_A | 2027-01 | 510000
DMA_A | 2027-01 | 505000
```

without knowing which value downstream consumers should use.

The current tables therefore answer a simple question:

> What is the current production forecast?

---

# Model Lineage

The project explicitly carries model identity across lifecycle stages.

```text
01 TRAIN
   |
   +-- candidate_run_id
            |
            v
02 VALIDATE
   |
   +-- validate exact run
            |
            v
03 PROMOTE
   |
   +-- promoted_model_version
            |
            v
04 DEPLOY
   |
   +-- deployed_model_version
            |
            v
05 FORECAST
   |
   +-- same verified model version
```

This prevents a class of production errors where:

```text
Model A is trained
Model B is deployed
Model C generates batch forecasts
```

without the discrepancy being noticed.

The pipeline verifies these relationships rather than assuming them.

---

# Databricks Task Values

Databricks task values provide the handoff between orchestration stages.

Examples include:

```text
candidate_run_id
promoted_model_version
deployed_model_version
forecast_model_version
```

This transformed the workflow from a manually coordinated collection of scripts into an automated pipeline.

For example:

```text
Task 01 produces Run ID
       ↓
Task 02 receives exact Run ID
       ↓
Task 03 promotes exact candidate
       ↓
Task 04 receives promoted version
       ↓
Task 05 receives deployed version
```

No manual copying of MLflow Run IDs or model versions is required during normal execution.

---

# Testing Strategy

The project contains approximately 50 automated tests covering the major system boundaries.

Run the full suite with:

```bash
pytest
```

Testing is performed locally before code is pushed to Git and pulled into Databricks.

---

## Candidate Validation Tests

Representative tests include:

```text
Strong candidate             -> PASS
Candidate worse than naive   -> FAIL
Validation WAPE > 15%        -> FAIL
Relative improvement < 5%    -> FAIL
Required metric missing      -> ERROR
Metric contains NaN / Inf    -> ERROR
```

The purpose is to ensure that protection logic is tested independently of live MLflow state.

---

## Model Promotion Tests

Promotion tests verify:

- completed runs may be promoted
- incomplete runs are rejected
- the correct registered model is targeted
- the `Champion` alias is assigned to the expected model version
- alias assignment is verified
- the promoted version is returned to downstream orchestration

Registry mutations are mocked locally.

---

## Deployment Tests

Deployment tests verify:

- deployed model versions are identified correctly
- missing served models cause failure
- updated served-entity configuration uses the requested version
- already-current endpoints follow the no-op path
- outdated endpoints trigger deployment
- endpoint versions are verified after update
- promoted model version must match `Champion`

---

## Forecasting Tests

Forecasting tests validate:

- future periods begin after historical observations
- forecast rows equal DMA count × forecast horizon
- latest static attributes are selected
- calendar features are constructed correctly
- recursive forecasts become future `Traffic_Lag_1`
- transaction assumptions remain consistent across horizons
- national forecasts equal summed DMA forecasts
- invalid forecast horizons are rejected
- batch inference model version matches `Champion`
- inference feature dtypes match the MLflow signature

---

# Failure Safety

A core design goal is to fail visibly when production assumptions are violated.

The system deliberately does **not** attempt to continue after important failures.

```text
Training Failure
      ↓
STOP

Candidate Validation Failure
      ↓
STOP

Promotion Failure
      ↓
STOP

Champion Verification Failure
      ↓
STOP

Serving Deployment Failure
      ↓
STOP

Serving Version Verification Failure
      ↓
STOP

MLflow Input Schema Failure
      ↓
STOP

Forecast Generation Failure
      ↓
STOP

Delta Write Verification Failure
      ↓
STOP
```

This behavior protects the integrity of downstream output.

The pipeline prioritizes explicit failure over silently producing forecasts from an invalid or inconsistent model state.

---

# Productionization Lessons

Moving the model from exploratory development into the final automated workflow exposed several practical MLOps issues.

Those issues influenced the final architecture.

---

## 1. Training Environment Is Not the Serving Environment

The model initially worked correctly in Databricks during training.

That did not guarantee that it could run inside Databricks Model Serving.

The isolated serving environment initially failed with:

```text
ModuleNotFoundError:
No module named 'databricks_forecasting'
```

The solution was not to modify the serving environment manually.

The custom application code was packaged with the MLflow model artifact.

This made the model self-contained enough for deployment.

---

## 2. A Model Artifact Includes More Than Model Weights

A production model requires:

```text
trained estimator
+
prediction wrapper
+
feature contract
+
runtime dependencies
+
custom source code
```

Simply serializing the XGBoost object would not reproduce the full prediction behavior required downstream.

---

## 3. Model Signatures Are Production Contracts

MLflow schema enforcement initially caused the batch forecasting task to fail because pandas produced:

```text
Year -> int64
```

while the registered model expected:

```text
Year -> int32
```

This was not treated as an inconvenience to disable.

It identified a real difference between training and production inputs.

The final architecture enforces the registered model dtypes immediately before prediction.

---

## 4. Registration Is Not Deployment

A successful registry promotion does not prove that a serving endpoint successfully loaded the new artifact.

The pipeline therefore treats:

```text
Model Promotion
```

and:

```text
Model Deployment
```

as separate verified state transitions.

---

## 5. Deployment Is Not Inference

Even after endpoint deployment, Task 05 verifies the model version it is using.

The architecture does not assume that successful deployment automatically guarantees the correct downstream batch inference version.

---

## 6. Job Outputs Should Be Explicit

Early development required manually copying Run IDs between steps.

The final system passes Run IDs and model versions through task values.

That improves:

- reproducibility
- traceability
- automation
- debugging
- lineage

---

## 7. Notebooks Should Orchestrate

The project began with larger notebook-oriented workflows.

The production refactor moved the durable logic into testable Python modules.

Databricks notebooks remain useful, but their role is now primarily orchestration.

---

## 8. Idempotency Matters

Re-running a production pipeline should not create unnecessary mutations.

The endpoint update logic therefore asks:

```text
Is the target model already deployed?
```

before submitting another deployment.

Likewise, production forecast tables use overwrite semantics rather than blindly appending duplicate predictions.

---

# Current Production Configuration

The project currently uses:

```text
MLflow Experiment
/Shared/marketing_forecasting
```

```text
Registered Model
workspace.default.marketing_traffic_forecast
```

```text
Production Alias
Champion
```

```text
Serving Endpoint
marketing-traffic-forecast
```

```text
Source Delta Table
workspace.default.synthetic_model_panel_v2
```

```text
DMA Forecast Table
workspace.default.future_traffic_forecast
```

```text
National Forecast Table
workspace.default.future_national_forecast
```

```text
Forecast Horizon
6 months
```

```text
Future Transaction Strategy
latest_observation
```

During final workflow validation, the serving endpoint successfully reached:

```text
Ready
```

with model:

```text
V10
```

---

# Configuration

Central configuration is maintained in:

```text
src/databricks_forecasting/config.py
```

Configuration includes:

```text
Registered model name
Model alias
Serving endpoint name
Source Delta table
Forecast Delta tables
Forecast horizon
Transaction assumption
Model feature list
Static feature list
Prediction target
Validation cutoff
Candidate hyperparameters
MLflow experiment
```

Central configuration prevents runtime settings from becoming scattered across orchestration notebooks and application modules.

---

# Current Limitations

The project intentionally stops at a complete working MLOps lifecycle rather than continuously increasing system complexity.

Several remaining limitations are explicit.

---

## Synthetic Modeling Data

The current implementation operates on synthetic data.

The architecture is intended to reproduce the shape of a production forecasting system while avoiding dependence on proprietary production datasets.

A real deployment would require validation against production source quality, definitions, refresh behavior, and operational dependencies.

---

## Recursive Forecast Uncertainty

The six-month forecast is recursive.

As forecasting moves further from the last observed month, predicted Traffic increasingly becomes an input into subsequent predictions.

Therefore:

```text
Horizon 1
```

contains less recursive uncertainty than:

```text
Horizon 6
```

Production monitoring should eventually evaluate accuracy separately by forecast horizon.

---

## Future Transactions

Transactions currently use the latest observed value as the future assumption.

This is intentionally simple.

If future Traffic is strongly sensitive to transaction dynamics, independently forecasting Transactions may improve longer-horizon performance.

---

## Static Future Characteristics

Operating, geographic, competitive, and customer-experience features are generally carried forward from the latest known observation.

This means the current system does not yet incorporate known future structural changes such as:

- planned store openings
- store closures
- future square-footage changes
- demographic projections
- competitive openings
- customer-experience initiatives
- major merchandising changes

---

## Forecast History

The production Delta tables represent only the current forecast snapshot.

Historical forecast vintages are not yet retained.

---

# Potential Future Directions

The architecture creates several natural paths for extension without requiring the core model lifecycle to be redesigned.

---

## 1. Forecast Vintage History

Add append-only historical forecast tables:

```text
workspace.default.future_traffic_forecast_history

workspace.default.future_national_forecast_history
```

Potential schema:

```text
Forecast_Run_ID
Forecast_Run_Timestamp
Model_Version
DMA
Forecast_Target_Month
Forecast_Horizon
Forecast
```

This would allow the system to answer:

> How much did the forecast for a given target month change between successive production runs?

For example:

```text
September run -> January forecast = 20.1M
October run   -> January forecast = 20.6M
November run  -> January forecast = 20.4M
```

That creates a useful measure of forecast stability and revision.

---

## 2. Production Forecast Accuracy

Once the target month becomes historical, stored forecasts can be compared against realized Traffic.

```text
Previous Forecast
        +
Realized Traffic
        ↓
Production Accuracy
```

Metrics could include:

- WAPE
- MAE
- RMSE
- mean error / bias
- DMA-level accuracy
- national accuracy
- horizon-specific accuracy
- seasonal accuracy

This would close the operational forecasting feedback loop.

---

## 3. Forecast Accuracy by Horizon

Because the model is recursive, accuracy should be measured independently for:

```text
Horizon 1
Horizon 2
Horizon 3
Horizon 4
Horizon 5
Horizon 6
```

This would quantify how quickly uncertainty increases as predicted Traffic enters subsequent feature calculations.

---

## 4. Feature and Prediction Drift

Monitor changes in:

```text
Traffic lag distributions
Transaction distributions
Store network characteristics
Competitive variables
CX measures
Prediction distributions
Residual distributions
```

Potential drift thresholds could then influence retraining decisions.

---

## 5. Transaction Forecasting

Replace:

```text
latest_observation
```

with a separate transaction forecasting process.

Future architecture:

```text
Transaction History
       ↓
Transaction Forecast Model
       ↓
Future Transactions
       ↓
Traffic Forecast Model
       ↓
Future Traffic
```

This would remove one of the largest simplifying assumptions in the current recursive workflow.

---

## 6. Known Future Business Inputs

Future model inputs could incorporate planning data such as:

- store openings
- store closures
- square-footage changes
- remodels
- competitive changes
- marketing investment
- promotional calendars
- customer-experience initiatives
- macroeconomic forecasts

This would shift the system from purely historical forecasting toward scenario-aware business forecasting.

---

## 7. Scenario Forecasting

The batch inference architecture could support scenarios such as:

```text
Base
Upside
Downside
```

Scenario inputs could vary:

```text
Transactions
Store Count
Competitive Density
CX Composite
Comp Sales Index
Macroeconomic assumptions
```

Outputs could then support planning instead of only point forecasting.

---

## 8. Expanded Champion / Challenger Evaluation

The current workflow evaluates an XGBoost candidate against a naive benchmark.

The governance layer could eventually compare:

```text
Current Champion
Candidate XGBoost
Candidate LightGBM
Candidate CatBoost
Statistical Forecast
Naive Benchmark
```

Promotion criteria could incorporate:

- overall WAPE
- forecast bias
- DMA stability
- seasonal performance
- horizon degradation
- inference latency
- model complexity

This would make promotion a multi-candidate selection process.

---

## 9. Automated Retraining Triggers

The current architecture can eventually support event-driven retraining.

Potential triggers include:

```text
Champion age > threshold
Forecast accuracy < threshold
Feature drift > threshold
Prediction drift > threshold
New data volume > threshold
Major business-calendar event
```

This would move retraining decisions beyond a purely scheduled process.

---

## 10. Production Notifications

Notifications could be added for events such as:

```text
Candidate trained
Candidate rejected
Candidate promoted
New Champion deployed
Endpoint deployment failed
Forecast published
Forecast accuracy degraded
Drift detected
```

Databricks Job notifications or downstream alerting integrations could support operational ownership.

---

## 11. CI/CD Expansion

The current development lifecycle is already:

```text
VS Code
  ↓
pytest
  ↓
Git
  ↓
Databricks
```

A future iteration could formalize this further with:

```text
GitHub Actions
Databricks Asset Bundles
Environment-specific configuration
Automated workflow deployment
```

Potential environments:

```text
Development
Staging
Production
```

The resulting workflow could become:

```text
Pull Request
     ↓
Automated Tests
     ↓
Merge
     ↓
Deploy Databricks Assets
     ↓
Integration Validation
     ↓
Production
```

---

## 12. Model and Data Monitoring Dashboard

A monitoring layer could report:

```text
Current Champion
Current deployed version
Last successful pipeline run
Latest forecast publication
Candidate vs Champion performance
Production WAPE
Forecast bias
Feature drift
Forecast revision
Endpoint status
```

This could provide a single operational view of the forecasting system.

---

# Design Principles

The final architecture follows several practical MLOps principles.

---

## Git Is the Source of Truth

Production application code originates from version-controlled source rather than workspace-only edits.

---

## Training Does Not Imply Promotion

A successfully trained model remains a candidate until it passes defined performance criteria.

---

## Promotion Does Not Imply Deployment

Registry state and serving state are verified independently.

---

## Deployment Does Not Imply Inference

Batch forecasting verifies model lineage before executing production inference.

---

## Model Identity Is Explicit

MLflow Run IDs and registered model versions flow between tasks rather than being manually copied or inferred.

---

## Model Contracts Are Enforced

Feature names, feature ordering, and dtypes are treated as part of the production model interface.

---

## Deployment Is Idempotent

An endpoint already serving the desired Champion does not receive an unnecessary deployment update.

---

## Forecast Outputs Represent Current Truth

Production Delta tables contain the latest successful forecast snapshot.

Historical forecasting analysis should be implemented separately rather than mixing forecast vintages into the current-state tables.

---

## Notebooks Orchestrate

Reusable business and modeling logic belongs in Python modules that can be independently tested.

---

## Failures Stop the Workflow

The pipeline does not silently bypass failed validation, deployment, schema validation, or output verification.

---

# End-to-End Lifecycle

The complete production lifecycle is:

```text
                           DATA
                            |
                            v
                    Feature Engineering
                            |
                            v
                    Candidate Training
                            |
                            v
                     MLflow Tracking
                            |
                            v
                    Naive Benchmark
                            |
                            v
                  Candidate Evaluation
                            |
                            v
                 Automated Validation
                      /           \
                   FAIL           PASS
                    |               |
                   STOP             v
                            Unity Catalog
                                  |
                                  v
                           Model Version
                                  |
                                  v
                           Champion Alias
                                  |
                                  v
                           Model Serving
                                  |
                                  v
                    Deployment Verification
                                  |
                                  v
                      Recursive Forecasting
                                  |
                                  v
                             Delta Lake
                         /              \
                        v                v
                DMA Forecast      National Forecast
```

---

# Summary

This project demonstrates how a forecasting model can be transformed from analytical code into a production-style machine learning system.

The completed implementation combines:

- data preparation
- feature engineering
- XGBoost forecasting
- time-based evaluation
- naive benchmarking
- MLflow experiment tracking
- MLflow model packaging
- model signatures
- automated validation gates
- Unity Catalog model registration
- Champion alias management
- Databricks Model Serving
- automated endpoint deployment
- deployment verification
- recursive batch inference
- Delta Lake persistence
- model lineage
- automated testing
- Git-based source control

The final automated workflow is:

```text
01 Train Candidate
        ↓
02 Validate Candidate
        ↓
03 Promote Candidate
        ↓
04 Update Endpoint
        ↓
05 Batch Forecast
```

Each stage has a clearly defined responsibility and failure boundary.

The result is a modular, testable, Git-controlled forecasting system where models must demonstrate measurable value before promotion, production deployment is independently verified, batch inference honors the registered model contract, and forecasts are published as authoritative Delta snapshots.

The system is complete as an end-to-end MLOps implementation while leaving clear extension points for forecast-vintage tracking, production accuracy monitoring, drift detection, transaction forecasting, scenario analysis, automated retraining, operational monitoring, and broader CI/CD automation.