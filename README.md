# CS611 Assignment 2 — Credit-Default ML Pipeline (Airflow + Docker)

A production-shaped ML pipeline that **trains, serves and governs** a loan-default
model across time, built on the Assignment-1 medallion datamart.

## Quick start
```bash
docker-compose build
docker-compose up
```
Open **http://localhost:8080** (admin / admin), enable the DAG
**`credit_default_ml_pipeline`** and let the backfill run (`catchup=True`,
Jan-2023 → Dec-2024).

## DAG — serving decoupled from governance
```
start
  -> run_data_pipeline -> data_pipeline_completed
       |-- run_inference -> run_monitoring -> generate_dashboard --|   (serving lane)
       |-- model_governance_train ---------------------------------|   (governance lane)
                                                                   -> end
```
* **Serving lane** scores every cohort with the *current* champion and writes a
  versioned, immutable predictions gold table, then monitoring + dashboards.
* **Governance lane** decides whether to (re)train for the *next* cycle and, when
  it does, writes a new dated artefact + registry row.

## Model governance SOP
| Trigger | Threshold | Action |
|---|---|---|
| Scheduled cadence | every 6 months | full 3-model bake-off + promote best OOT AUC |
| Performance (AUC) | < 0.70 | retrain + re-select |
| Stability (PSI) | ≥ 0.25 | retrain on recent data |
| Stability (PSI) | 0.10 – 0.25 | investigate; watch |
| Calibration drift | actual ≫ predicted | recalibrate / refresh |

* **Bootstrap:** a Logistic Regression champion is promoted on the initial anchor
  **2024-06-01** (right after the first OOT window).
* **Refresh:** on a trigger, the full bake-off (LogReg / RandomForest / XGBoost)
  runs on a rolling window and the best OOT-AUC model is promoted.

## Model bank / registry
* `model_bank/credit_model_<date>.pkl` — dated, immutable artefacts
* `model_bank/champion_latest.pkl` — pointer to the live champion (serving)
* `model_bank/model_registry.csv` — auditable version history (date, action, champion, metrics)

## Layout
| path | purpose |
|------|---------|
| `dags/ml_pipeline_dag.py` | branched Airflow DAG |
| `scripts/` | data pipeline, governance trainer, inference, monitoring, dashboard |
| `utils/model_utils.py` | Spark-agnostic ML core (modeling table, training, governance, monitoring) |
| `utils/data_processing_*` | A1 medallion bronze/silver/gold |
| `data/` · `datamart/` · `model_bank/` · `artifacts/` | inputs, gold tables, models, dashboards |

## Leakage controls
* **Temporal**: features taken at application month `M − 6`; label at maturity `M`.
* **Train/test**: scaling + encoding fit inside the sklearn Pipeline on training folds only.
* **Target**: post-application loan-performance fields blocked from the feature store by assert.
