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
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    },
}

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)


def check_submission_format(
    submission_file: Path, test_file: Path, subtask: str
) -> bool:
    """
    Checks whether the submission format is correct.
    test dataset to compare the ids of your submission.

    It checks whether:
        (i) the file is a valid jsonl file
        (ii) all columns has not empty values
        (iii) the labels are correct for the subtask.
        (iv) labels are not always the same (model working wrongly)
        (v) the number of samples in the submission match those in the test set
        (vi) the submission ids match the test ids

        Args:
            submission_file (Path): submission file with the predictions.
            test_file (Path): test dataset to compare the ids of your submission.
            subtask (str): either subtask_1 or subtask_2.

        Returns:
            bool: True if the format is correct and False otherwise.
    """
    # The submission file is a valid JSONL
    try:
        submission = pd.read_json(
            submission_file, lines=True, orient="records"
        )[["id", "label"]]
    except ValueError:
        logging.error(
            "File is not a valid jsonl file: {}".format(submission_file)
        )
        return False

    # Not empty columns
    for column in ["id", "label"]:
        if submission[column].isna().any():
            logging.error(
                "NA value in file {} in column {}".format(
                    submission_file, column
                )
            )
            return False

    # Labels within the expected set of labels
    if not submission["label"].isin(LABELS[subtask]).all():
        logging.error("Unknown Label in file {}".format(submission_file))
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

    # Labels are not always the same.
    # Not an error, but warns about potentially wrong predictions.
    if len(submission["label"].unique()) == 1:
        label = submission["label"][0]
        logging.warning(
            f"Your model is predicting always {label}."
            "This isn't a format error, but your model may be functioning incorrectly."
        )

    # Same number of samples in submission than in the test set
    test_df = pd.read_json(test_file, lines=True, orient="records")
    len_sub = len(submission)
    len_test = len(test_df)
    if len_sub != len_test:
        logging.error(
            f"The number of predicted examples ({len_sub}) "
            f"does not match with the test set ({len_test})"
        )
        return False

    # Submission ids match the test ids
    if not len(
        set(submission["id"].tolist()).intersection(set(test_df["id"].tolist()))
    ) == len(test_df):
        logging.error(
            "There is a mismatch between the ids of team and the test set."
        )
        return False

    logging.info("Your submission has a valid format :)")
    return True
