# Databricks Forecasting MLOps Pipeline

An end-to-end forecasting and MLOps project built to demonstrate how a machine learning model can move from training through validation, model governance, deployment, and production batch inference using Databricks, MLflow, Unity Catalog, Model Serving, Delta Lake, Git, and automated testing.

The project forecasts future traffic at the DMA level and produces both DMA-level and national six-month forecasts.

The focus of the project is not only model development. It is the complete operational lifecycle surrounding the model:

- reproducible training
- baseline evaluation
- automated candidate validation
- controlled model promotion
- Unity Catalog model versioning
- Champion alias management
- automated serving endpoint deployment
- deployment verification
- recursive batch forecasting
- Delta table outputs
- automated testing
- Git-based source control and development

---

## Architecture

```text
                       Historical Delta Table
                                |
                                v
                    +-----------------------+
                    | 01 Train Candidate    |
                    +-----------------------+
                                |
                         MLflow Run ID
                                |
                                v
                    +-----------------------+
                    | 02 Validate Candidate |
                    +-----------------------+
                          |             |
                        FAIL           PASS
                          |             |
                         STOP           v
                                +----------------------+
                                | 03 Promote Candidate |
                                +----------------------+
                                           |
                                  Registered Version
                                  Champion Alias
                                           |
                                           v
                                +----------------------+
                                | 04 Update Endpoint   |
                                +----------------------+
                                           |
                                  Ready Model Serving
                                           |
                                    Deployed Version
                                           |
                                           v
                                +----------------------+
                                | 05 Batch Forecast    |
                                +----------------------+
                                           |
                          +----------------+----------------+
                          |                                 |
                          v                                 v
              DMA Forecast Delta Table          National Forecast Delta Table