# Orchestration — the same 8 stages, two ways

`backend/app/ml/training.py` runs the eight training stages as function calls in
one process. This directory runs the *same stages, calling the same code*, under
Flyte and under Airflow.

Both were executed here. Both produced **ROC AUC 0.7408** — the same number the
FastAPI path produces, because the ML never changed:

```
run-20260811-230758   flyte     0.7408
run-20260811-230925   airflow   0.7408
```

That is the payoff of keeping `app/ml/` free of HTTP. FastAPI was one caller;
an orchestrator is just another one.

## Running them

**Flyte** — no cluster needed, `flytekit` executes locally:

```bash
pip install flytekit pyarrow
python orchestration/flyte/hotel_ml_workflow.py
# or:
pyflyte run orchestration/flyte/hotel_ml_workflow.py training_workflow --cv_folds 3
```

**Airflow** — needs its own environment; it pins dependencies hard enough to
downgrade the backend's pandas/numpy if installed alongside:

```bash
python -m venv .airflow-venv
.airflow-venv/bin/pip install apache-airflow==3.3.0 pandas scikit-learn pyarrow

export AIRFLOW_HOME=$PWD/.airflow
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/orchestration/airflow/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

.airflow-venv/bin/airflow db migrate
.airflow-venv/bin/airflow dags reserialize     # `dags list` reads the DB, not the folder
.airflow-venv/bin/airflow dags test hotel_ml_training
```

## The differences that actually matter

### 1 · How data moves between tasks

**This is the difference you feel every day.**

Flyte passes typed values. A task returns a `pd.DataFrame` and the next task
takes a `pd.DataFrame`; Flyte serialises it to Parquet, stores it, and hands it
over:

```python
@task
def validate(frame: pd.DataFrame, target: str) -> pd.DataFrame: ...
```

Airflow's XCom is a row in the metadata database. It is sized for identifiers,
not datasets, so a 6,000-row frame does not belong in it. The idiom is to write
a file and pass the **path**:

```python
@task
def validate(raw_path: str) -> str:
    frame = pd.read_parquet(raw_path)
    ...
    frame.to_parquet(path)
    return str(path)          # only the path crosses the boundary
```

Airflow can do the Flyte thing via a custom XCom backend, but you have to build
it. Out of the box, you manage storage yourself.

### 2 · When the graph is decided

The Flyte `@workflow` body is **not ordinary Python at run time**. It executes
once to *compile* a graph: calling a task returns a promise, and the edges come
from which promise feeds which task. So this cannot work:

```python
if cv_folds >= 2:            # a Python branch cannot compile into a static graph
    cross_validate(...)
```

The skip decision has to live *inside* the task instead — which is why the
Flyte version returns `{"skipped": True}` rather than not running.

Airflow resolves its graph per run, so branching is real:

```python
@task.branch
def should_cross_validate() -> str:
    return "cross_validate" if CONFIG["cv_folds"] >= 2 else "skip_cross_validation"
```

Verified in the run above:

```
should_cross_validate     success
cross_validate            success
skip_cross_validation     skipped     ← genuinely not executed
register                  success     ← still ran, thanks to trigger_rule
```

That `trigger_rule="none_failed_min_one_success"` is required. The default rule
needs *every* upstream to succeed, and a branch leaves one of them skipped — so
without it, `register` would never run. This trips up everyone once.

### 3 · Type checking, and when it fails

Flyte checks types at compile time. This version of the workflow was rejected
before a single task ran:

```
ValueError: Type of Generic List type is not supported,
Only generic univariate typing.List[T] type is supported.
```

The cause was `importances: list` — Flyte needs a concrete element type to build
a schema for the value crossing the boundary. `List[dict]` fixed it.

Airflow would have accepted the bare annotation and failed later, at runtime,
on serialisation — after the expensive training task had already run.

### 4 · Caching

Flyte hashes a task's inputs. A repeat call with the same arguments skips
execution and reuses the stored output:

```python
@task(cache=True, cache_version="1.0")
def ingest(rows: int, seed: int) -> pd.DataFrame: ...
```

Airflow has no equivalent. Re-running a DAG re-runs the tasks; you build your
own idempotency and caching.

### 5 · Scheduling

Airflow is a scheduler that runs DAGs. It is declared on the DAG itself:

```python
@dag(schedule="0 3 * * *", catchup=False, max_active_runs=1)
```

`catchup=False` matters: with it `True`, a DAG whose `start_date` is months back
immediately queues one run per missed interval. Flyte does schedules too, via
launch plans, but they are a separate concept rather than a DAG property.

### Summary

| | Flyte | Airflow |
|---|---|---|
| Data between tasks | typed values, engine-managed | file paths through XCom |
| Type checking | compile time | runtime, if at all |
| Branching | not in the workflow body | `@task.branch`, real skips |
| Caching | built in, by input hash | roll your own |
| Scheduling | launch plans | first-class on the DAG |
| Local run | `python file.py`, no infra | needs a metadata DB |
| Isolation | one container per task by default | shared worker unless configured |
| Job market | niche | **dominant** |

## Which to learn

**Airflow, if you are optimising for employability** — it is in far more job
postings, and the concepts (DAGs, operators, XCom, trigger rules, backfill)
transfer to Dagster and Prefect.

**Flyte, if you are optimising for ML work** — typed data passing and free
caching genuinely fit training pipelines better, and re-running an expensive
step you have already computed is the daily annoyance it removes.

Both appear in the Expedia JD. Airflow first if you only do one.

## What this is not

Both versions are single-machine demos. Production adds:

- **Remote execution** — Flyte on Kubernetes, Airflow with Celery/Kubernetes executors
- **Real storage** — S3/GCS instead of `/tmp`
- **Secrets and connections** — managed, not hardcoded config dicts
- **Backfills** — re-running historical windows after a bug fix
- **Alerting** — on failure, on SLA miss
- **Data-aware scheduling** — trigger on a dataset landing, not the clock

Also note what neither version does: **the eight stages here have no per-stage
status writes**. `training.py` reports live progress into a run record so the UI
can animate. An orchestrator owns that state itself — its own UI shows task
status — so pushing it into `run.json` too would be duplicating the orchestrator's
job. In a real migration you would drop the custom stage tracking and read the
orchestrator's API instead.
