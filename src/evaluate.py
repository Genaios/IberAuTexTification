"""
Module for the official evaluation.
"""

from functools import partial
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import classification_report, f1_score


def evaluate_submissions(
    submissions_path: Path,
    ground_truth_path: Path,
    subtask: str,
    output_file: Path,
) -> pd.DataFrame:
    """
    Evaluates all the submissions in the submissions path.
    This is the official evaluation code that the organizers will use
    to rank submissions. Computes f1 per-class and macro-averaged,
    accuracy, a classification report and confidence intervals of the
    macro-averaged f1.

    Args:
        submissions_path (Path): base path for submissions.
        ground_truth_path (Path): base path for ground truths.
        subtask (str): either subtask_1 or subtask_2.
        output_file (Path): file to store the ranking and results.

    Returns:
        pd.DataFrame: a ranking of the submissions.
    """
    results: Dict[str, List] = {
        "team": [],
        "run": [],
        "all_metrics": [],
        "mf1": [],
        "mf1_cinterval": [],
    }
    ground_truth_df = pd.read_json(
        next(ground_truth_path.glob(f"{subtask}/truth.jsonl")),
        lines=True,
        orient="records",
    )

    for run in submissions_path.glob(f"*/{subtask}/*"):
        team = str(run.parents[1]).split("/")[-1]
        run_name = run.stem
        run_df = pd.read_json(run, lines=True, orient="records")
        if len(run_df) != len(ground_truth_df):
            print(
                f"The number of predicted examples does not match with the reference: {team}"
            )
            continue
        if not len(
            set(run_df.index.tolist()).intersection(
                set(ground_truth_df.index.tolist())
            )
        ) == len(ground_truth_df):
            print(
                f"There is a mismatch between the ids of team {team} and the ground truth."
            )
            continue

        results["team"].append(team)
        results["run"].append(run_name)

        run_df = run_df.join(
            ground_truth_df,
            on="id",
            how="left",
            lsuffix="_pred",
            rsuffix="_true",
        )

        y_true = run_df["label_true"]
        y_pred = run_df["label_pred"]
        results["all_metrics"].append(
            classification_report(
                y_true=y_true,
                y_pred=y_pred,
                digits=4,
                output_dict=True,
            )
        )
        results["mf1"].append(
            f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        )

        mf1_cinterval = bootstrap(
            data=[y_true, y_pred],
            statistic=partial(f1_score, average="macro"),
            n_resamples=100,
            paired=True,
            confidence_level=0.95,
            method="basic",
        )
        results["mf1_cinterval"].append(
            (
                mf1_cinterval.confidence_interval.low,
                mf1_cinterval.confidence_interval.high,
            )
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mf1", ascending=False)
    results_df.to_csv(output_file, sep="\t", index=False)
    return results_df
