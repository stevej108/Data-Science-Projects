# Data Science Projects

A portfolio of applied data science, machine learning, causal inference, forecasting, optimization, and AI projects developed across professional, consulting, and independent work.

The projects in this repository demonstrate the application of statistical modeling and machine learning to real-world business problems — from marketing measurement and demand forecasting to medical-device signal processing, optimization, customer analytics, computer vision, and recommendation systems.

My focus is not simply on building models, but on developing **end-to-end analytical solutions** that connect:

**data engineering → exploratory analysis → statistical modeling → machine learning → validation → interpretation → decision support**

---

## Core Capabilities

This repository demonstrates experience across a broad range of data science disciplines:

* **Machine Learning** — XGBoost, LightGBM, Random Forests, ensemble methods, gradient boosting, classification, and regression
* **Forecasting & Time Series** — traffic forecasting, demand forecasting, Prophet, lag analysis, seasonality, and external-regressor modeling
* **Causal Inference & Marketing Measurement** — Bayesian Structural Time Series (BSTS), Difference-in-Differences, panel regression, counterfactual modeling, placebo testing, and incrementality measurement
* **Statistical Modeling** — hypothesis testing, regression, fixed effects, time-series diagnostics, uncertainty estimation, and model validation
* **Optimization** — linear programming, facility-location optimization, and operational resource allocation
* **Unsupervised Learning** — clustering, segmentation, and customer profiling
* **Deep Learning** — neural networks, computer vision, and NLP
* **Explainable AI** — feature importance, SHAP, model diagnostics, and interpretable reporting
* **Data Engineering** — ETL pipelines, REST APIs, data validation, transformation, aggregation, and cloud integration
* **Analytics Engineering & Reporting** — reproducible Python pipelines, automated reporting, visualization, and business-facing decision support

---


# Featured Projects

| Project | Focus |
|---|---|
| [Marketing Incrementality & Causal Impact](Marketing_Incrementality/) | Causal inference, BSTS, DiD, panel regression, marketing measurement |
| [Traffic Forecasting](Forecasting/) | XGBoost, time-series forecasting, feature engineering, model validation |
| [Productivity Optimization](Productivity%20Optimization/) | Operational analytics, productivity measurement, optimization |



## [Marketing Incrementality & Causal Impact](Marketing_Incrementality/)

A comprehensive marketing-measurement framework designed to determine whether incremental marketing investment generates incremental store traffic across U.S. Designated Market Areas (DMAs).

Rather than relying on a single methodology, the project progressively evaluates the marketing intervention using multiple statistical and causal-inference approaches.

### Methods

* Exploratory data analysis
* Seasonality and temporal diagnostics
* Cross-correlation and lag analysis
* DMA and time fixed-effects panel regression
* Difference-in-Differences
* Bayesian Structural Time Series / Causal Impact
* Counterfactual traffic estimation
* Posterior uncertainty analysis
* Placebo testing
* Robustness analysis
* Comparable-store traffic translation
* DMA-level spend explanatory-power modeling
* Market opportunity scoring

The project moves from descriptive relationships toward increasingly controlled estimates of marketing impact and ultimately integrates the evidence into DMA-level decision support.

**[View Project →](Marketing_Incrementality/)**

---

## [Traffic Forecasting](Forecasting/)

An end-to-end machine-learning forecasting framework designed to predict future store traffic at both DMA and national levels.

The pipeline integrates historical traffic with external and operational predictors including store characteristics, marketing activity, weather, customer-experience information, seasonality, and engineered time-series features.

### Workflow

* Multi-source data ingestion and validation
* DMA standardization
* Feature engineering
* Model-readiness validation
* Machine-learning model training and tuning
* Out-of-sample forecast validation
* DMA-level error analysis
* Feature-importance analysis
* Future DMA forecasts
* National forecast aggregation
* Year-over-year forecasting
* Automated HTML reporting

The project emphasizes not only predictive performance but also geographic consistency, model diagnostics, and business interpretability.

**[View Project →](Forecasting/)**

---

## [Productivity Optimization](Productivity%20Optimization/)

An applied analytics project focused on evaluating productivity and identifying opportunities for operational improvement through data-driven analysis.

The project demonstrates the use of structured analytical workflows to move from operational data through measurement, modeling, visualization, and actionable decision support.

**[View Project →](Productivity%20Optimization/)**

---

# Optimization & Operations Research

## Facility Location Optimization for Emergency Services

Developed a facility-location optimization model using **PuLP and CBC** to identify optimal EMS and Fire Service locations within Montgomery County based on geographic demand.

The project demonstrates the use of mathematical optimization to support public-service resource allocation and minimize geographic service constraints.

**Techniques:** Linear Programming, Facility Location Optimization, PuLP, CBC

---

# Forecasting & Time-Series Modeling

## Retail Demand Forecasting

Developed product-level and category-level demand forecasting models using **XGBoost and Prophet**.

The forecasting framework incorporated external regressors including:

* Seasonality
* Competitor pricing
* Weather
* Promotions
* Historical sales behavior

An interactive Plotly-based dashboard was also developed to evaluate forecast behavior at the store level.

**Techniques:** XGBoost, Prophet, Time-Series Forecasting, Feature Engineering, Plotly

---

## Grocery Store Sales Forecasting

Built predictive models to evaluate expected performance across grocery-store locations.

The analysis supported resource-allocation decisions by identifying higher-traffic and lower-traffic locations and providing a quantitative framework for operational planning.

**Techniques:** Exploratory Data Analysis, Regression, Machine Learning, Forecasting

---

# Medical Device & Life Sciences Analytics

## Diagnostic Device Signal Forecasting

Developed predictive models for electrode-signal behavior in a molecular diagnostic device.

Exploratory analysis, statistical modeling, Random Forest, and LightGBM techniques were used to identify signal patterns associated with false-negative outcomes and improve device signal interpretation.

**Techniques:** Random Forest, LightGBM, Signal Analytics, Classification, Statistical Modeling

---

## AI/ML Signal Bleed Algorithm

Developed a lightweight machine-learning algorithm to identify and mitigate off-target electrode signal bleeding in a medical diagnostic device.

The solution incorporated Random Forest, XGBoost, ensemble modeling, and data visualization techniques and achieved a substantial reduction in false-positive signals attributable to signal bleed.

**Techniques:** Random Forest, XGBoost, Ensemble Learning, Signal Processing, Classification

---

## Small Molecule Predictive Analytics & ETL

Additional projects explore predictive analytics and data-engineering workflows supporting small-molecule analysis, including batch-release modeling and ETL pipeline development.

These projects demonstrate the application of data science to scientific and manufacturing workflows where model reliability, data quality, and reproducibility are particularly important.

---

# Computer Vision

## Leaf Disease Detection

Developed a deep-learning computer-vision model to distinguish healthy and diseased leaves.

The model achieved approximately **95% classification accuracy**, with SHAP-based explainability used to investigate the image regions and features contributing to individual predictions.

**Techniques:** Deep Learning, Computer Vision, Image Classification, SHAP, Explainable AI

---

# Financial Services & Risk Modeling

## Loan Approval Predictive Modeling

Developed an ensemble machine-learning framework to evaluate applicant credit risk and estimate the likelihood of loan default using historical applicant and financial characteristics.

**Techniques:** Classification, Ensemble Learning, Risk Modeling, Feature Engineering

---

## Credit Card Customer Churn

Built a predictive classification model designed to identify customers at elevated risk of credit-card churn or attrition.

The project focused on producing a robust and generalizable model while identifying the features most strongly associated with customer attrition.

**Techniques:** Classification, Churn Modeling, Feature Importance, Predictive Analytics

---

## Banking Customer Segmentation

Applied unsupervised machine-learning techniques to segment banking customers based on financial history, demographics, and customer interactions.

The resulting segments were used to identify opportunities for improved customer engagement and more targeted product-marketing strategies.

**Techniques:** Clustering, Customer Segmentation, Unsupervised Learning, Customer Analytics

---

# Marketing & Customer Analytics

## Mixed-Media Advertising Analysis

Evaluated the effectiveness of multiple advertising channels against target customer demographics.

The analysis examined both advertising performance and the customer characteristics most associated with conversion across different media channels.

**Techniques:** Predictive Modeling, Marketing Analytics, Conversion Analysis, Customer Profiling

---

## Channel Mix Media Analytics

Applied statistical analyses to identify channel-level associative relationship between media channels and traffic.

The statistical analysis provided a detailed view of channel-level impacts on key business drivers to develop targeted marketing strategies.

**Techniques:** Marketing Analytics, Descriptive Statistics, Inferential Statistics

---

## Marketing Customer Segmentation

Applied clustering and predictive modeling to identify distinct customer groups and determine which customer characteristics were associated with greater conversion potential.

The resulting segmentation provided a framework for more targeted marketing campaigns and customer acquisition strategies.

**Techniques:** Clustering, Segmentation, Predictive Modeling, Marketing Analytics

---

# Recommendation Systems

## Amazon Product Recommendation System

Developed a recommendation framework using customer purchase behavior, browsing history, segmentation, and demographic information.

The model was designed to identify relevant products for individual customers and demonstrate how machine-learning techniques can support personalized product recommendations.

**Techniques:** Recommendation Systems, Customer Segmentation, Collaborative/Behavioral Modeling, Machine Learning

---

# Natural Language Processing

## NLP Chatbot Training

Developed an NLP-based chatbot training framework using an LSTM deep-learning architecture and GloVe word representations.

The project explored how sequence models and word embeddings can be used to generate more natural interactions between users and conversational systems.

**Techniques:** NLP, LSTM, Deep Learning, GloVe, Word Embeddings

---

# Healthcare Predictive Modeling

## Cirrhosis / Liver Disease Classification

Applied deep-learning techniques to patient data to classify stages of liver disease.

The project demonstrates the application of supervised learning to multiclass healthcare prediction and the challenges associated with classification in clinically oriented datasets.

**Techniques:** Deep Learning, Multiclass Classification, Healthcare Analytics

---

# Data Engineering

## End-to-End ETL Pipeline

Developed an ETL workflow to retrieve real-time data from a RESTful API, perform cleaning, filtering, aggregation, and statistical analysis, and load processed outputs into **Google Cloud Platform (GCP)** for downstream analytics and machine-learning applications.

**Techniques:** Python, REST APIs, ETL, Data Transformation, Cloud Computing, GCP

---

# Additional Exploratory Work

## NFL Play Exploratory Data Analysis

Conducted exploratory analysis of NFL pre-snap behavior with the objective of identifying patterns useful for play prediction.

The project explores high-dimensional sports data and the challenge of extracting predictive structure from large feature spaces.

**Techniques:** Exploratory Data Analysis, Feature Engineering, Predictive Modeling, Sports Analytics

---

# Repository Organization

The repository contains a combination of full analytical pipelines, Jupyter notebooks, and technical case studies.

```text
Data-Science-Projects/
│
├── Forecasting/
│   └── End-to-end traffic forecasting pipeline
│
├── Marketing_Incrementality/
│   └── Marketing measurement and causal inference
│
├── Productivity Optimization/
│   └── Operational analytics and optimization
│
├── Facility_Location_Optimization_using_PuLP_and_CBC.ipynb
├── Retail_Demand_Forecasting_using_Prophet.ipynb
├── Signal Forecasting for Molecular Diagnostics.ipynb
├── Computer Vision for Leaf Detection.ipynb
├── Ensemble Learning for Loan Approval Predictions.ipynb
├── Credit_Card_Users_Churn_Predictive Modeling.ipynb
├── FinTech Bank Customer Clustering for Resource Optimization.ipynb
├── Amazon Recommender System Development.ipynb
├── MarCom Targeted Advertising Modeling.ipynb
├── NLP_LLM Chatbot Training.ipynb
│
└── Additional technical case studies and documentation
```

---

# Technical Toolkit

Across these projects, I have worked with tools and methodologies including:

**Languages & Core Analytics**

`Python` · `SQL` · `pandas` · `NumPy` · `SciPy`

**Machine Learning**

`scikit-learn` · `XGBoost` · `LightGBM` · `Random Forest` · `Ensemble Learning`

**Statistics & Causal Inference**

`statsmodels` · `Panel Regression` · `Difference-in-Differences` · `BSTS` · `Counterfactual Modeling` · `Hypothesis Testing`

**Forecasting**

`Prophet` · `Gradient Boosting` · `Time-Series Analysis` · `Lag Analysis` · `Seasonality Modeling`

**Deep Learning & AI**

`Neural Networks` · `LSTM` · `Computer Vision` · `NLP` · `Recommendation Systems`

**Explainability & Visualization**

`SHAP` · `Matplotlib` · `Plotly` · `Statistical Diagnostics`

**Optimization & Engineering**

`PuLP` · `CBC` · `REST APIs` · `ETL` · `GCP`

---

# Approach

Across projects, I emphasize several principles:

1. **Start with the business problem.** Modeling choices should be driven by the decision the analysis is intended to support.

2. **Build reproducible analytical workflows.** Data preparation, modeling, validation, and reporting should be structured so analyses can be rerun and extended.

3. **Separate prediction from causality.** Predictive relationships are not automatically interpreted as causal effects.

4. **Validate beyond a single metric.** Aggregate model performance can conceal important differences across markets, customer groups, time periods, or other segments.

5. **Prioritize interpretability.** Statistical sophistication is most valuable when results can be translated into understandable business implications.

6. **Treat models as decision-support tools.** The ultimate objective is not simply model accuracy, but better-informed decisions.

---

# About This Repository

This repository represents a selected portfolio rather than a complete history of my professional work.

Some projects originated from consulting or professional analytical work and have been adapted where necessary to protect proprietary information. Underlying business datasets, confidential information, credentials, and certain generated outputs are intentionally excluded.

The projects are intended to demonstrate both the **breadth of techniques** I have worked with and the **depth required to move from raw data to a defensible analytical conclusion**.
