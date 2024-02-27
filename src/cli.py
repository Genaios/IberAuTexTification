from pathlib import Path

import typer
from typing_extensions import Annotated

from . import models
from .cli_utils import (
    infer_subtask,
    load_task_dataset,
    read_json,
    save_predictions,
)
from .evaluate import evaluate_submissions
from .format_checker import check_submission_format

app = typer.Typer()


@app.command()
def run_experiment(
    config_file: Annotated[
        Path,
        typer.Option(
            help="JSON file with model configs.",
            exists=True,
            file_okay=True,
            resolve_path=True,
        ),
    ],
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the dataset folder.",
            exists=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    team_name: Annotated[str, typer.Option(help="Team name")],
    do_train: Annotated[bool, typer.Option(help="Fit the model")] = True,
    do_predict: Annotated[bool, typer.Option(help="Evaluate the model")] = True,
    output_path: Annotated[
        Path,
        typer.Option(
            help="Path to store predictions.",
            exists=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("evaluation_data/submissions"),
) -> None:
    """
    Fit models and predicts a dataset, saving the predictions
    into disk, ready to be evaluated with the `evaluate` endpoint.

    You can use this endpoint to run the baselines or to run
    your own models (if you implement them in `models.py`), both
    for subtasks 1 and 2.

    Args:
        - config_file (Path): config file for the experiments.
        - dataset_path (Path): folder containing your dataset in jsonl format.
        - team_name (str): the name of your team.
        - do_train (bool): train the models listed in config file.
            `do_train` needs a train dataset within the `dataset_path` folder.
            Pass `--no-do-train` to avoid training.
        - do_predict (bool): predict with the models listed in the config file.
            `do_predict` needs a test dataset within the `dataset_path` folder.
            Pass `--no-do-predict` to avoid predicting.
        - output_path (Path): path to store the predictions. By default is the
            path used by the official evaluation script.
    """
    assert do_predict or do_train, "`do_train` or `do_predict` must be True."
    config = read_json(config_file)
    dataset = load_task_dataset(dataset_path)
    subtask = infer_subtask(dataset["train"])
    output_path = output_path / team_name / subtask
    output_path.mkdir(parents=True, exist_ok=True)
    for model_name, model_args in config.items():
        model_class = getattr(models, model_args["class"])
        model = model_class(**model_args["params"])
        if do_train:
            model.fit(dataset["train"])
        if do_predict:
            preds = model.predict(dataset["test"])
            output_file = (output_path / model_name).with_suffix(".jsonl")
            save_predictions(output_file, dataset["test"]["id"], preds)


@app.command()
def evaluate(
    subtask: Annotated[
        str,
        typer.Option(help="Subtask, either `subtask_1` or `subtask_2`"),
    ],
    submissions_path: Annotated[
        Path,
        typer.Option(
            help="Path containing submissions.",
            exists=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("evaluation_data/submissions"),
    ground_truth_path: Annotated[
        Path,
        typer.Option(
            help="Path containing ground truths.",
            exists=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("evaluation_data/ground_truth"),
    output_file: Annotated[
        Path,
        typer.Option(
            help="File name to store the ranking dataframe.",
            resolve_path=True,
        ),
    ] = Path("./ranking_results.tsv"),
) -> None:
    """
    Evaluates submissions for subtasks 1 and 2. This is the
    official evaluation code that the organizers will use
    to rank submissions.

    The ranking of the submissions is either printed in terminal
    and stored in `output_file` in tsv format.

    Args:
        subtask (str): either subtask_1 or subtask_2.
        submissions_path (Path): base path for submissions.
        ground_truth_path (Path): base path for ground truths.
        output_file (Path): file to store the ranking and results.
    """
    ranking_df = evaluate_submissions(
        submissions_path, ground_truth_path, subtask, output_file
    )
    print(ranking_df)


@app.command()
def check_format(
    submission_file: Annotated[
        Path,
        typer.Option(
            help="File with predictions.",
            file_okay=True,
            resolve_path=True,
        ),
    ],
    test_file: Annotated[
        Path,
        typer.Option(
            help="File with the test dataset.",
            file_okay=True,
            resolve_path=True,
        ),
    ],
    subtask: Annotated[
        str,
        typer.Option(help="Subtask, either `subtask_1` or `subtask_2`"),
    ],
) -> None:
    """
    Checks whether the submission format is correct.

    Args:
        submission_file (Path): submission file with the predictions.
        test_file (Path): test dataset to compare the ids of your submission.
        subtask (str): either subtask_1 or subtask_2.
    """
    check_submission_format(submission_file, test_file, subtask)


if __name__ == "__main__":
    app()
