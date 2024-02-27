"""
Module for modeling.

Here are defined classes for the baselines, but participants can implement
their models in this module and get for free the CLI endpoints:
train, predict, and evaluate.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .cli_utils import set_seed

SEED = 13
set_seed(SEED)


class ClassificationModel(ABC):
    """
    Base class to implement classification models.
    """

    def __init__(
        self,
        model_params: Dict,
        tokenizer_params: Dict,
        training_params: Dict,
        inference_params: Dict,
    ):
        self.model_params = model_params
        self.tokenizer_params = tokenizer_params
        self.training_params = training_params
        self.inference_params = inference_params

    @abstractmethod
    def fit(self, train_dataset: Dataset) -> "ClassificationModel": ...

    @abstractmethod
    def predict(self, test_dataset: Dataset) -> List[str]: ...


class LogisticRegressionBagOf(ClassificationModel):
    """
    Logistic Regression model using bag-of-ngrams (words/chars) as features.
    """

    def __init__(
        self,
        model_params: Dict,
        tokenizer_params: Dict,
        training_params: Dict,
        inference_params: Dict,
    ):
        super().__init__(
            model_params, tokenizer_params, training_params, inference_params
        )
        self.word_vectorizer = CountVectorizer(
            analyzer="word",
            ngram_range=tuple(tokenizer_params["word"]["ngram_range"]),
            max_features=tokenizer_params["word"]["max_features"],
        )

        self.char_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=tuple(tokenizer_params["char"]["ngram_range"]),
            max_features=tokenizer_params["char"]["max_features"],
        )

        self.pipeline = make_pipeline(
            FeatureUnion(
                [("word", self.word_vectorizer), ("char", self.char_vectorizer)]
            ),
            StandardScaler(with_mean=False),
            LogisticRegression(random_state=SEED),
        )

    def fit(self, train_dataset: Dataset) -> ClassificationModel:
        self.pipeline.fit(train_dataset["text"], train_dataset["label"])
        return self

    def predict(self, test_dataset: Dataset) -> List[str]:
        return self.pipeline.predict(test_dataset["text"])


class SymantoDualEncoder(ClassificationModel):
    """
    Symanto zero and few shot models.

    This model will be run internally by the organizers.
    Participants will get import errors since the SDK
    is not available out of the company.
    """

    def __init__(
        self,
        model_params: Dict,
        tokenizer_params: Dict,
        training_params: Dict,
        inference_params: Dict,
    ):
        from symanto_dec import DeClassifier

        super().__init__(
            model_params, tokenizer_params, training_params, inference_params
        )

        self.model = DeClassifier(
            self.model_params["pretrained_model_name_or_path"],
            label2text=self.model_params["label2text"],
            batch_size=self.training_params.get("batch_size", 1),
            epochs=self.training_params.get("epochs", 4),
            lr=self.training_params.get("lr", 2e-5),
        )

    def _get_shots_per_label(
        self, dataset: Dataset, shots: int
    ) -> Tuple[List[str], List[str]]:
        n_labels: defaultdict = defaultdict(lambda: 0)
        texts, labels = [], []
        for example in dataset:
            label = example["label"]
            if n_labels[label] < shots:
                texts.append(example["text"])
                labels.append(label)
                n_labels[label] += 1
        return texts, labels

    def fit(self, train_dataset: Dataset) -> ClassificationModel:
        if self.training_params["shots"] > 0:
            texts, labels = self._get_shots_per_label(
                train_dataset, self.training_params["shots"]
            )
            self.model.fit(
                texts, labels, strategy=self.training_params["strategy"]
            )
        else:
            self.model.fit()

        return self

    def predict(self, test_dataset: Dataset) -> List[str]:
        assert (
            self.model.encoder is not None
        ), "Do `fit` to initialize the encoder."
        return self.model.predict(test_dataset["text"])


class HuggingFaceClassifier(ClassificationModel):
    """
    HuggingFace models for sequence classification tasks
    using either encoder o encoder-decoder models.
    """

    def __init__(
        self,
        model_params: Dict,
        tokenizer_params: Dict,
        training_params: Dict,
        inference_params: Dict,
    ):
        super().__init__(
            model_params, tokenizer_params, training_params, inference_params
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            **self.model_params
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_params["pretrained_model_name_or_path"],
            **self.tokenizer_params
        )

    def fit(self, train_dataset: Dataset) -> ClassificationModel:
        tok_dataset = train_dataset.select_columns(["text", "label"])
        tok_dataset = tok_dataset.map(
            lambda batch: self.tokenizer(batch, truncation=True),
            input_columns=["text"],
            batched=True,
            remove_columns=["text"],
        )
        tok_dataset = tok_dataset.map(
            lambda label: {"label": self.model_params["label2id"][label]},
            input_columns=["label"],
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(**self.training_params)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tok_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model()
        return self

    def predict(self, test_dataset: Dataset) -> List[str]:
        tok_dataset = test_dataset.select_columns(["text"])
        tok_dataset = tok_dataset.map(
            lambda batch: self.tokenizer(batch, truncation=True),
            input_columns=["text"],
            batched=True,
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        inference_args = TrainingArguments(**self.inference_params)
        trainer = Trainer(
            model=self.model,
            args=inference_args,
            train_dataset=tok_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        preds = trainer.predict(tok_dataset)

        # Some encoder-decoder models returns predictions as a tuple.
        if isinstance(preds.predictions, tuple):
            preds = preds.predictions[0].argmax(-1)
        else:
            preds = preds.predictions.argmax(-1)

        pred_labels = [
            self.model_params["id2label"][str(pred)] for pred in preds
        ]
        return pred_labels
