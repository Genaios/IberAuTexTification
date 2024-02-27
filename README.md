<h1 align="center">IberAuTexTification 👩🏻🤖</h1>

<p align="center">
    <a href="LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-CC_BY_NC_ND_4.0-green">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0-green">
    </a>
    <img alt="Static Badge" src="https://img.shields.io/badge/Languages-en%2Ces%2Cgl%2Ceu%2Cca-blue">
    <img alt="Static Badge" src="https://img.shields.io/badge/Subtasks-detection%2Cattribution-yellow">
    <img alt="Static Badge" src="https://img.shields.io/badge/Award-500%E2%82%AC-green">
    <img alt="Static Badge" src="https://img.shields.io/badge/LLMs-GPT4%2CGPT3.5%2CCommand%2CJurassic%2CLLaMa2%2CMixtral-purple">
</p>

<h3 align="center"><b>Automated Text Identification on Languages of the Iberian Peninsula</b></h3>
</br>

The **Iber AuTexTification: Automated Text Identification on Languages of the Iberian Peninsula** shared task will take place as part of **IberLEF 2024**, the **6th Workshop on Iberian Languages Evaluation Forum at the SEPLN 2024 Conference**, which will be held in **Valladolid**, **Spain** on the **26th of September**, **2023**.


IberAuTexTification is the second version of the AuTexTification at IberLEF 2023 shared task (Sarvazyan et al., 2023). We extend our previous task in three dimensions: more models, more domains and more languages from the Iberian Peninsula (in a multilingual fashion), aiming to build more generalizable detectors and attributors. In this task, participants must develop models that exploit clues about linguistic form and meaning to identify automatically generated texts from a wide variety of models, domains, and languages. We plan to include LLMs like GPT-3.5, GPT-4, LLaMA, Coral, Command, Falcon, MPT, among others. New domains like essays, or dialogues, and cover the most prominent languages from the Iberian Peninsula: Spanish, Catalan, Basque, Galician, Portuguese, and English (in Gibraltar).  

For all the information about the shared task (description, how to download the dataset, constraints, etc.), please, refer to the [webpage](https://sites.google.com/view/iberautextification/home).

## Award

To foster engagement and reward dedication, we will award the best participant in each subtask with **500€** sponsored by [Genaios](https://genaios.ai/).

We hope for your participation and good luck in the competition! 🍀


## Subtasks

A novelty from this edition is to detect in a **multilingual** (languages from the **Iberian peninsula** such as Spanish, English, Catalan, Gallego, Euskera, and Portuguese), **multi-domain** (news, reviews, emails, essays, dialogues, wikipedia, wikihow, tweets, emails, etc.), and **multi-model** (GPT, LLaMA, Mistral, Cohere, Anthropic, MPT, Falcon, etc.) setup, whether a text has been automatically generated or not, and, if generated, identify the model that generated the text. There is only one dataset containing all the languages for each subtask, instead of different language tracks per subtask as in the previous edition.

The two subtasks of this edition of the shared task are:

**Subtask 1: Human or Generated**: participants will be provided a text, and they will have to determine whether the text has been automatically generated or not. To encourage models to learn features that generalize to new writing styles, three domains will be used for training, and two different domains for testing. The dataset will contain the following columns: *id*, *text*, *domain*, *language*, *model*, and *label*.


**Subtask 2: Model attribution**: Participants will be provided an automatically generated text, and they will have to determine what model generated it.
The datasets will have the following columns: *id*, *text*, *domain*, *language*, and *label*.


The first subtask is a binary classification task with two classes: 👩🏻 and 🤖. The models used to generate text are instructed LLMs from very different providers like OpenAI, Amazon Bedrock, Anthropic, Cohere, AI21, Google Vertex AI, Meta, etc. The datasets have been generated using [TextMachina](https://github.com/Genaios/TextMachina), a tool to create MGT datasets through a wide variety of prompts, controling classical biases present in this kind of datasets.

The datasets will include texts from domains like essays, news, social media (tweets, forums, dialogues), wikipedia, wikihow, etc. Texts from uncontrolled domains as extracted from the OSCAR (Abadji et al., 2022) and Colossal Cleaned Multilingual Common Crawl (Raffel, 2019) will be included too.

# What is this repo for?
This repo contains code to run the baselines, evaluate your predictions and check the format of your submissions for both subtasks. 

The code is prepared with extensibility in mind, so you can use it as basis to develop your own models and get some functionalities for free as CLI endpoints or config handling.

## Run Baselines

To run the baselines first you need to place the datasets of the subtasks into the folders `subtask_1` and/or `subtask_2` within the `task_datasets` folder. These datasets must be in **jsonl** format and you can include `train`, `validation`, and `test` partitions depending on your use case. We already provided dummy datasets within this repo, but you will need to place the task dataset into the folders once you download it.

Once the data is prepared, you can run the baselines using the `run-experiment` of the [CLI script](src/cli.py) as follows:

```bash
python -m src.cli run-experiment \
--config-file ./etc/subtask_1_baselines.json \
--dataset-path task_datasets/subtask_1 \
--team-name baselines \
--do-train
--do-predict
```

You need to specify the configuration file with all the parameters of the models you want to run (see [etc/subtask_1_baselines.json](etc/subtask_1_baselines.json) to know more about this), the dataset folder, and your team name to prepare the predictions accordingly to the format expected by the official evaluation script. You can also decide whether doing training or prediction (`--do-train/--no-do-train` and `--do-predict/--no-do-predict`), but by default, both `--do-train` and `--do-predict` are True.

If `--do-predict` is passed, the predictions will be stored in `evaluation_data/submissions/[team-name]`, ready to be evaluated using the official evaluation code.

As for the baselines, you can implement your own models in [models.py](src/models.py), define their arguments in a config file, and run the `run-experiment` endpoint to perform your experimentation.

## Evaluation

The code to compute the official evaluation scores and rank the submissions can be found in the [evaluate.py](src/evaluate.py) module. It computes f1 per-class and macro-averaged, accuracy, a classification report and confidence intervals of the macro-averaged f1.

Before running the evaluation script, you will need to put your labeled test set, in **jsonl** format, within the corresponding folder in [evaluation_data/ground_truth](evaluation_data/ground_truth).

Once the prediction files from your model are stored in `evaluation_data/submissions/[team-name]` (this is automatically done if you ran the `run-experiment` endpoint), you can run the `evaluate` endpoint of the [CLI script](src/cli.py) as follows:

```bash
python -m src.cli evaluate \
--subtask subtask_1 \
--output-file ranking_subtask_1.tsv
```

The ranking of the submissions will be printed in the terminal and stored in `output_file` in tsv format.

For testing purposes, we provided the prediction files of several baselines for dummy datasets in [evaluation_data/submissions/baselines](evaluation_data/submissions/baselines), along with the ground truths of these dummy datasets in [evaluation_data/ground_truth](evaluation_data/ground_truth).

## Format checking

To ensure that your submission for a subtask has the correct format as expected by the evaluation code, you can run the `check_format` endpoint of the [CLI script](src/cli.py) as follows:

```bash
python -m src.cli check-format \
--submission-file evaluation_data/submissions/baselines/subtask_1/mt5-large.jsonl \
--subtask subtask_1
```

If there is any issue with your submission, it will log the specific error that needs to be solved.
Please, check that your submissions have the correct format before submitting the final predictions to the organizers.

# FAQ

**Q: Are there any modeling constraints in this task?**

Yes, the constraints are the following.

1) Publicly available pretrained models from the literature can be used. However, you are only allowed to use text derived from the training data. That is, data augmentation, further self-supervised pre-training, or other techniques that involve the usage of additional text must be done only with text derived from the training data. 

2) The usage of knowledge bases, lexicons and other structured data resources is also allowed.

3) Usage of data from one subtask in the other subtask is not allowed.

**Q: How many submissions can we submit?**

3 submissions are allowed per team for each subtask.

**Q: Should we do all subtasks or just one of them?**

Participants are free to participate in any of the two subtasks.

# Organizers

- Areg Sarvazyan (areg.sarvazyan@genaios.ai) - Genaios, Valencia, Spain
- José Ángel González (jose.gonzalez@genaios.ai) - Genaios, Valencia, Spain
- Marc Franco-Salvador (marc.franco@genaios.ai) - Genaios, Valencia, Spain
- Francisco Rangel (francisco.rangel@genaios.ai) - Genaios, Valencia, Spain
- Paolo Rosso (prosso@dsic.upv.es) - Universitat Politècnica de València, Valencia, Spain

# Social

Google groups: [https://groups.google.com/g/iberautextification](https://groups.google.com/g/iberautextification)

Slack channel: [https://join.slack.com/t/iberautextification/shared_invite/zt-2c28ezgwy-lHHM6ASHnqLY2YQ8mlPgdQ&sa=D&sntz=1&usg=AOvVaw1oYekQiDZ0_C_-N79NtReu](https://join.slack.com/t/iberautextification/shared_invite/zt-2c28ezgwy-lHHM6ASHnqLY2YQ8mlPgdQ&sa=D&sntz=1&usg=AOvVaw1oYekQiDZ0_C_-N79NtReu)