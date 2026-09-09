# Marketing Incrementality Analysis

An end-to-end marketing measurement framework designed to evaluate the relationship between incremental marketing investment and store traffic across U.S. Designated Market Areas (DMAs).

The project combines exploratory analysis, panel regression, difference-in-differences, Bayesian structural time series (BSTS), robustness testing, comp-traffic analysis, and DMA-level opportunity scoring to move from descriptive relationships toward estimates of incremental marketing impact.

## Project Objectives

The analysis is designed to answer several related questions:

* How is marketing spend associated with store traffic across markets and over time?
* Are observed relationships driven by seasonality, market differences, or broader time trends?
* Does incremental marketing investment correspond with incremental traffic after controlling for DMA and time effects?
* How would traffic likely have performed in the absence of incremental marketing intervention?
* How consistent are estimated marketing effects across markets?
* How do estimated traffic effects translate into comparable-store traffic performance?
* Which DMAs demonstrate the strongest combination of marketing responsiveness, model confidence, and future investment opportunity?

Rather than relying on a single model, the project uses multiple analytical approaches to evaluate the marketing intervention from complementary perspectives.

---

## Analytical Framework

The project is organized as a sequential analytical pipeline.

### 1. Data Preparation

Raw traffic, marketing spend, and store data are loaded, standardized, cleaned, and aggregated into analysis-ready DMA/month panels.

Key steps include:

* Data validation and cleaning
* DMA name standardization
* Store-to-DMA mapping
* Monthly aggregation
* Marketing treatment identification
* Feature engineering
* Panel construction
* Data quality and coverage diagnostics

### 2. Exploratory Data Analysis

Exploratory analysis evaluates the underlying structure of the traffic and marketing data before causal modeling.

Analysis includes:

* Traffic distributions
* Marketing spend distributions
* DMA-level comparisons
* Store-count and market-size diagnostics
* Correlation analysis
* Spend-versus-traffic relationships
* Outlier detection
* Panel balance and coverage
* Variance inflation factor diagnostics

### 3. Seasonality Analysis

Traffic and marketing data are evaluated for temporal structure and seasonal patterns.

Methods include:

* Monthly traffic and spend trends
* Year-over-year traffic analysis
* Seasonal indices
* STL decomposition
* Autocorrelation (ACF)
* Partial autocorrelation (PACF)

These diagnostics help distinguish underlying temporal structure from potential marketing-related effects.

### 4. Cross-Correlation Analysis

Cross-correlation analysis evaluates the temporal relationship between marketing spend and traffic.

The analysis compares both raw and adjusted time series and includes:

* Overall spend/traffic correlation
* Lagged correlations
* Rolling correlations
* Cross-correlation functions (CCF)
* Detrended and deseasonalized comparisons
* DMA-level correlations

Comparing raw and adjusted relationships helps determine whether apparent lag structures persist after accounting for common trends and seasonality.

### 5. Panel Regression

Panel regression evaluates whether incremental marketing spend is associated with incremental traffic after controlling for persistent market differences and common time effects.

Four specifications are compared:

1. Pooled OLS
2. DMA fixed effects
3. Time fixed effects
4. DMA + Time fixed effects

The DMA + Time fixed-effects specification provides the primary controlled panel estimate.

Model diagnostics include:

* Marketing-spend coefficients
* Model fit comparisons
* Residual diagnostics
* Actual-versus-predicted traffic
* DMA-level model comparisons

### 6. Difference-in-Differences

Difference-in-differences (DiD) provides an additional framework for comparing treated and control markets before and after marketing intervention.

The analysis includes:

* Treatment definition
* Treatment/control comparisons
* Parallel-trends diagnostics
* Treatment-effect estimation
* Continuous-spend specifications

This provides a complementary estimate of marketing-associated changes while explicitly evaluating the assumptions underlying the DiD framework.

### 7. Bayesian Structural Time Series / Causal Impact

Bayesian structural time series models estimate counterfactual traffic for treated DMAs.

For each market, the model estimates:

> What would traffic likely have been if incremental marketing intervention had not occurred?

Observed post-intervention traffic is compared with the model-estimated counterfactual to calculate:

* Incremental traffic
* Relative traffic lift
* Posterior uncertainty
* Probability of a positive effect
* DMA-level contribution to aggregate impact

This framework provides both an estimated treatment effect and an explicit representation of uncertainty.

### 8. Model Validation

The BSTS framework is subjected to additional validation before the estimated effects are used for reporting.

Validation includes:

* Posterior diagnostics
* Counterfactual validation
* Control-market validation
* Model coverage diagnostics
* Pre-intervention predictive performance
* DMA-level validation summaries

The objective is to distinguish statistically estimated effects from effects supported by sufficiently reliable model behavior.

### 9. Robustness Testing

Additional robustness analyses test whether the primary conclusions remain stable under alternative assumptions and specifications.

Tests include:

* Placebo analysis
* Counterfactual placebo validation
* DMA heterogeneity analysis
* Marketing-spend relationship analysis
* Directional consistency checks
* Alternative aggregation and comparison views

These tests provide additional context around the stability and reliability of the estimated marketing effects.

### 10. Comp-Traffic Translation

BSTS traffic effects are translated into comparable-store traffic terms to provide a more operationally interpretable measure of marketing impact.

The implied counterfactual comp is calculated as:

```text
Implied Counterfactual Comp =
((1 + Actual Comp) / (1 + BSTS Relative Lift)) - 1
```

Estimated incremental comp traffic is then:

```text
Implied Incremental Comp =
Actual Comp - Implied Counterfactual Comp
```

This allows each DMA to be evaluated in terms of how marketing-associated traffic lift or drag contributed to observed comparable-store performance.

### 11. DMA Spend Explanatory Power

DMA-level regression analysis evaluates how strongly variation in marketing spend explains variation in traffic within individual markets.

Outputs include:

* DMA-specific spend coefficients
* Partial R²
* Model coverage
* Leave-one-out sensitivity analysis
* Spend responsiveness versus explanatory power

This analysis helps distinguish markets where spend is merely correlated with traffic from markets where spend provides meaningful incremental explanatory value.

### 12. DMA Opportunity Analysis

The final stage integrates evidence across the analytical pipeline to evaluate DMA-level marketing opportunity.

The framework combines measures of:

* Historical performance
* Marketing responsiveness
* Model confidence
* Spend explanatory power
* Current investment profile
* Cross-model alignment

The resulting opportunity scores provide a decision-support framework for identifying markets where additional marketing investment may warrant further consideration.

---

## Project Pipeline

The analysis is implemented as a sequence of Python modules:

```text
01_load_data.py
        │
        ▼
02_clean_data.py
        │
        ▼
02a_standardize_dmas.py
        │
        ▼
03_feature_engineering.py
        │
        ▼
04_eda.py
        │
        ▼
05_seasonality.py
        │
        ▼
06_cross_correlation.py
        │
        ▼
07_panel_regression.py
        │
        ▼
08_difference_in_differences.py
        │
        ▼
09_causal_impact.py
        │
        ▼
10_causal_impact_validation.py
        │
        ▼
11_causal_impact_robustness.py
        │
        ▼
12_reporting.py
        │
        ▼
13_comp_traffic_analysis.py
        │
        ▼
14_bsts_comp_comparison.py
        │
        ▼
15_dma_spend_regression.py
        │
        ▼
16_dma_opportunity_analysis.py
```

`run_pipeline.py` provides orchestration for the analytical workflow.

---

## Repository Structure

```text
Marketing_Incrementality/
│
├── 01_load_data.py
├── 02_clean_data.py
├── 02a_standardize_dmas.py
├── 03_feature_engineering.py
├── 04_eda.py
├── 05_seasonality.py
├── 06_cross_correlation.py
├── 07_panel_regression.py
├── 08_difference_in_differences.py
├── 09_causal_impact.py
├── 10_causal_impact_validation.py
├── 11_causal_impact_robustness.py
├── 12_reporting.py
├── 13_comp_traffic_analysis.py
├── 14_bsts_comp_comparison.py
├── 15_dma_spend_regression.py
├── 16_dma_opportunity_analysis.py
│
├── config.py
├── run_pipeline.py
├── .gitignore
│
├── data/                   # Local analysis data (excluded from Git)
│
└── outputs/                # Generated models/results (excluded from Git)
```

Raw data and generated analytical outputs are intentionally excluded from the public repository.

---

## Key Methodologies

The project incorporates several complementary statistical and causal-inference techniques:

| Method                          | Primary Purpose                                                 |
| ------------------------------- | --------------------------------------------------------------- |
| Exploratory Data Analysis       | Understand distributions, coverage, outliers, and relationships |
| Seasonality Analysis            | Identify temporal and recurring traffic patterns                |
| Cross-Correlation               | Evaluate contemporaneous and lagged spend/traffic relationships |
| Panel Regression                | Control for DMA and time-specific effects                       |
| Difference-in-Differences       | Compare treated and control market changes                      |
| Bayesian Structural Time Series | Estimate counterfactual traffic and incremental lift            |
| Placebo Testing                 | Evaluate robustness of estimated causal effects                 |
| Comp-Traffic Translation        | Express estimated lift in operational comp terms                |
| DMA-Level Regression            | Measure spend explanatory power within individual markets       |
| Opportunity Scoring             | Integrate evidence into market-level decision support           |

The use of multiple methodologies is intentional. No individual model is treated as definitive; conclusions are evaluated based on consistency across methods, diagnostics, uncertainty, and robustness testing.

---

## Technology Stack

The project is implemented in Python and uses statistical, econometric, Bayesian, and visualization libraries for:

* Data manipulation and transformation
* Panel-data analysis
* Statistical modeling
* Bayesian inference
* Time-series analysis
* Causal inference
* Visualization
* Automated HTML reporting

Primary analytical packages include tools from the Python scientific computing ecosystem such as pandas, NumPy, SciPy, statsmodels, matplotlib, and Bayesian modeling libraries.

---

## Outputs

The pipeline generates a structured set of analytical artifacts, including:

```text
outputs/
├── figures/
├── tables/
├── models/
├── reports/
├── logs/
└── comp_traffic/
```

Generated outputs include:

* EDA figures and diagnostic tables
* Seasonality diagnostics
* Cross-correlation analyses
* Panel regression results
* Difference-in-differences results
* BSTS posterior models
* Counterfactual estimates
* Model validation diagnostics
* Placebo and robustness analyses
* DMA-level impact estimates
* Comp-traffic comparisons
* DMA spend explanatory-power results
* DMA opportunity scores
* HTML analytical reports

These artifacts are generated locally and are not stored in the public repository.

---

## Interpretation Philosophy

Marketing incrementality is inherently difficult to estimate from observational data.

For that reason, this project intentionally avoids interpreting simple correlation as causal impact. Instead, evidence is progressively evaluated through:

```text
Descriptive Relationships
        ↓
Temporal Diagnostics
        ↓
Controlled Panel Models
        ↓
Quasi-Experimental Analysis
        ↓
Counterfactual Modeling
        ↓
Validation
        ↓
Robustness Testing
        ↓
Cross-Model Synthesis
        ↓
Decision Support
```

A market demonstrating a positive correlation between spend and traffic is therefore not automatically considered responsive to marketing.

Greater confidence requires alignment across multiple forms of evidence, including controlled estimates, counterfactual performance, model diagnostics, uncertainty, and robustness tests.

---

## Data Availability

The datasets used in this analysis are not included in the repository.

The public repository contains the analytical framework and source code only. Raw data, processed datasets, generated model artifacts, and analysis outputs are excluded through `.gitignore`.

This separation allows the analytical architecture and methodology to be reviewed without distributing underlying business data.

---

## Limitations

Results from observational marketing measurement should be interpreted as estimates rather than experimental proof of causality.

Potential limitations include:

* Non-random allocation of marketing spend
* Unobserved market-level confounders
* Changes in competitive or macroeconomic conditions
* Differences in store maturity and market composition
* Limited pre- or post-intervention observations for some DMAs
* Sensitivity of counterfactual models to control-market selection
* Potential differences in treatment timing and intensity across markets

The use of fixed effects, DiD, BSTS counterfactual modeling, validation, placebo testing, and robustness analysis is intended to reduce these risks, but cannot eliminate them entirely.

---

## Purpose

This project demonstrates an end-to-end approach to marketing incrementality measurement that extends beyond descriptive reporting.

The analytical workflow moves from:

**data engineering → exploratory analysis → statistical modeling → causal inference → validation → business interpretation → market prioritization**

with the goal of producing marketing measurement that is statistically defensible, operationally interpretable, and useful for decision-making.
