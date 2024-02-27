"""
Module to check format errors.
Adapted from: https://github.com/mbzuai-nlp/SemEval2024-task8
"""

import logging
from pathlib import Path

import pandas as pd

LABELS = {
    "subtask_1": {"human", "generated"},
    "subtask_2": {
        "cohere.command-text-v14",
        "gpt-3.5-turbo-instruct",
        "gpt-4",
        "ai21.j2-ultra-v1",
        "meta-llama/Llama-2-70b-chat-hf",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    },
}

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)
COLUMNS = ["id", "label"]


def check_submission_format(path: Path, subtask: str) -> bool:
    """
    Checks whether the submission format is correct.

    It checks whether:
        (i) the file is a valid jsonl file
        (ii) all columns has not empty values
        (iii) the labels are correct for the subtask.

    Args:
        submission_file (Path): submission file with the predictions.
        subtask (str): either subtask_1 or subtask_2.

    Returns:
        bool: True if the format is correct and False otherwise.
    """
    try:
        submission = pd.read_json(path, lines=True, orient="records")[
            ["id", "label"]
        ]
    except ValueError:
        logging.error("File is not a valid jsonl file: {}".format(path))
        return False
    for column in COLUMNS:
        if submission[column].isna().any():
            logging.error(
                "NA value in file {} in column {}".format(path, column)
            )
            return False

    if not submission["label"].isin(LABELS[subtask]).all():
        logging.error("Unknown Label in file {}".format(path))
        logging.error(
            "Unique Labels in the file are {}".format(
                submission["label"].unique()
            )
        )
        logging.error(
            (
                f"Check your labels are correct for {subtask}."
                f"The allowed labels for this subtask are: {LABELS['subtask']}"
            )
        )
        return False
    logging.info("Your submission has a valid format :)")
    return True
