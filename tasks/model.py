"""
This file contains the tasks that are used to download, train and evaluate the model.
"""

from pathlib import Path

from datasets import Dataset
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from typing_extensions import Annotated
from union import Artifact, Deck, Resources, current_context, task

from containers import container_image

# Define Artifact Specifications
FineTunedImdbModel = Artifact(name="fine_tuned_Imdb_model")


# ---------------------------
# download model
# ---------------------------
@task(
    container_image=container_image,
    cache=True,
    cache_version="1",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_model(model_name: str) -> FlyteDirectory:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    working_dir = Path(current_context().working_directory)
    saved_model_dir = working_dir / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save model and tokenizer
    model.save_pretrained(
        saved_model_dir,
    )
    tokenizer.save_pretrained(saved_model_dir)

    return FlyteDirectory(saved_model_dir)


# ---------------------------
# full fine-tune model
# ---------------------------
@task(
    container_image=container_image,
    requests=Resources(cpu="4", mem="12Gi", gpu="1"),
)
def train_model(
    model_dir: FlyteDirectory,
    train_dataset: FlyteFile,
    val_dataset: FlyteFile,
    epochs: int = 3,
) -> Annotated[FlyteDirectory, FineTunedImdbModel]:

    import pandas as pd
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # Download the model directory locally
    local_model_dir = model_dir.download()

    # Load model and tokenizer from saved directory
    model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

    # Load datasets from CSV and limit for faster training during tutorial
    train_df = pd.read_csv(train_dataset.download()).sample(n=500, random_state=42)
    val_df = pd.read_csv(val_dataset.download()).sample(n=100, random_state=42)

    # Convert DataFrames to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Tokenization and training logic
    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenizer_function)
    tokenized_val_dataset = val_dataset.map(tokenizer_function)

    training_args = TrainingArguments(
        output_dir="./results", num_train_epochs=epochs, evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

    trainer.train()

    # Save the trained model
    output_dir = Path(current_context().working_directory) / "trained_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # return FlyteDirectory(output_dir)
    return FineTunedImdbModel.create_from(output_dir)


# ---------------------------
# evaluate model
# ---------------------------
@task(
    container_image=container_image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="12Gi", gpu="1"),
)
def evaluate_model(trained_model_dir: FlyteDirectory, test_dataset: FlyteFile) -> dict:
    import numpy as np
    import pandas as pd
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # Download the model directory locally
    local_model_dir = trained_model_dir.download()

    # Load model and tokenizer from the saved directory
    model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

    # Load and prepare test dataset
    # test_df = pd.read_csv(test_dataset.download())
    test_df = pd.read_csv(test_dataset.download()).sample(n=100, random_state=42)

    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_test_dataset = test_dataset.map(tokenize_function)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()

    return eval_results
