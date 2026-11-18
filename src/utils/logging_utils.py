"""Logging utilities for experiments."""

from pathlib import Path
from loguru import logger

from src.utils.currency import format_cost, record_cost_usd


def log_experiment_config(
    run_dir: Path,
    model: str,
    dataset_path: Path,
    prompt_path: Path,
    sample_size: int | None,
    temperature: float,
    think: bool | str,
    num_predict: int,
    results_path: Path,
    metrics_path: Path,
    condition: str = "vanilla",
    concurrency: int | None = None,
    retrieval_method: str | None = None,
    retrieval_top_k: int | None = None,
    sample_random: bool = False,
    sample_seed: int | None = None,
    chat_template_kwargs: dict | None = None,
):
    """Log a run's configuration."""

    lines = [
        "=" * 56,
        "EXPERIMENT CONFIG",
        "=" * 56,
        "",
        f"Run ID              : {run_dir.name}",
        f"Condition           : {condition}",
        f"Model               : {model}",
        "",
        f"Dataset Path        : {dataset_path}",
        f"Prompt Path         : {prompt_path}",
        "",
        f"Sample Size         : {sample_size}",
        f"Sample Random       : {sample_random}",
        f"Sample Seed         : {sample_seed}",
        "",
        f"Temperature         : {temperature}",
        f"Think Mode          : {think}",
        f"Template Kwargs     : {chat_template_kwargs}",
        f"Num Predict         : {num_predict}",
        f"Concurrency         : {concurrency}",
    ]

    if retrieval_method is not None:
        lines.extend(
            [
                "",
                f"Retrieval Method    : {retrieval_method}",
                f"Retrieval Top-K     : {retrieval_top_k}",
            ]
        )

    lines.extend(
        [
            "",
            f"Results Path        : {results_path}",
            f"Metrics Path        : {metrics_path}",
            f"Run Directory       : {run_dir}",
            "",
            "=" * 56,
        ]
    )

    logger.info("\n{}", "\n".join(lines))


def log_summary(metrics: dict):
    """Log a run's final metrics."""

    lines = [
        "=" * 56,
        "EXPERIMENT SUMMARY",
        "=" * 56,
        "",
        f"Run ID              : {metrics['run_id']}",
        f"Condition           : {metrics.get('condition', 'vanilla')}",
        f"Model               : {metrics['model']}",
        f"Provider            : {metrics['provider']}",
        f"Retrieval Method    : {metrics.get('retrieval_method')}",
        f"Retrieval Top-K     : {metrics.get('retrieval_top_k')}",
        "",
        f"Questions Processed : {metrics['total']}",
        f"Correct             : {metrics['correct']}",
        f"Skipped             : {metrics['skipped']}",
        f"Accuracy            : {metrics['accuracy']:.4f}",
        "",
        f"Total Tokens        : {metrics['total_tokens']}",
        f"Input Tokens        : {metrics['total_input_tokens']}",
        f"Output Tokens       : {metrics['total_output_tokens']}",
        "",
        f"Estimated Cost      : {format_cost(record_cost_usd(metrics))}",
        f"Eval Total Time     : {metrics['eval_total_seconds']:.2f}s",
        f"Total Request Time  : {metrics['total_request_seconds']:.2f}s",
        "",
        (
            "Avg Input Tokens    : "
            f"{metrics['avg_input_tokens']:.2f}"
        ),
        (
            "Avg Output Tokens   : "
            f"{metrics['avg_output_tokens']:.2f}"
        ),
        (
            "Avg Request Time    : "
            f"{metrics['avg_request_seconds']:.2f}s"
        ),
        "",
        f"Results File        : {metrics['results_path']}",
        f"Metrics File        : {metrics['metrics_path']}",
        "",
        "=" * 56,
    ]

    logger.info("\n{}", "\n".join(lines))
