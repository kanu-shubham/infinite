"""Airflow DAG that runs a weekly LoRA fine-tuning → merge → deploy pipeline.

Requires the Kubernetes Provider and an in-cluster Airflow scheduler with
permission to launch Jobs in the ``ml-training`` namespace.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="lora_finetune_weekly",
    description="Weekly LoRA fine-tuning pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "lora", "llm"],
) as dag:
    image = "ghcr.io/ORG/lora-finetune:train-latest"
    gpu_resources = {"requests": {"nvidia.com/gpu": "1"}, "limits": {"nvidia.com/gpu": "1"}}

    train = KubernetesPodOperator(
        task_id="train",
        name="lora-train",
        namespace="ml-training",
        image=image,
        cmds=["lora-finetune"],
        arguments=["train", "--config", "/configs/qlora_llama3_8b.yaml"],
        container_resources=gpu_resources,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    merge = KubernetesPodOperator(
        task_id="merge",
        name="lora-merge",
        namespace="ml-training",
        image=image,
        cmds=["lora-finetune"],
        arguments=[
            "merge",
            "--adapter",
            "/outputs/qlora-llama3-8b-instruct/adapter",
            "--output",
            "/outputs/qlora-llama3-8b-instruct/merged",
        ],
        is_delete_operator_pod=True,
        get_logs=True,
    )

    deploy = KubernetesPodOperator(
        task_id="rollout_serve",
        name="kubectl-rollout",
        namespace="ml-training",
        image="bitnami/kubectl:1.30",
        cmds=["kubectl"],
        arguments=["rollout", "restart", "deployment/lora-finetune-serve", "-n", "ml-serving"],
        is_delete_operator_pod=True,
    )

    train >> merge >> deploy
