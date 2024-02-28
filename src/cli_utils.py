"""
Module with CLI utils.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset


def set_seed(seed: int = 13) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_json(path: Path) -> Dict:
    with open(path, "r") as fr:
        return json.load(fr)


def load_task_dataset(path_or_name: Path) -> Dataset:
    dataset = load_dataset("json", data_dir=path_or_name)
    return dataset


def save_predictions(
    output_file: Path, ids: List[str], preds: List[str]
) -> None:
    df = pd.DataFrame({"id": ids, "label": preds})
    df.to_json(output_file, lines=True, orient="records")


def infer_subtask(dataset: Dataset) -> str:
    if len(set(dataset["label"]).union({"human", "generated"})) == 2:
        return "subtask_1"
    return "subtask_2"
