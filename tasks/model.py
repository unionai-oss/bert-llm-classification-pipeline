'''
This file contains the tasks that are used to download, train and evaluate the model.
'''

from union import task, Resources, current_context, Deck
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from pathlib import Path
from containers import container_image
from transformers import BertForSequenceClassification


# ---------------------------
# download model
# ---------------------------
@task(
    container_image=container_image,
    cache=True,
    cache_version="v3",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_model(model: str) -> FlyteDirectory:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    working_dir = Path(current_context().working_directory)
    model_cache_dir = working_dir / "model_cache"

    AutoTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
    AutoModelForSequenceClassification.from_pretrained(model, cache_dir=model_cache_dir)
    return model_cache_dir

# ---------------------------
# train model
# ---------------------------
@task(
    container_image=container_image,
    requests=Resources(cpu="4", mem="12Gi", gpu="1"),
)
def train_model(model_name: str, 
                dataset_cache_dir: FlyteDirectory,
                model_cache_dir: FlyteDirectory,
                epochs: int = 3
    ) -> BertForSequenceClassification:
    from datasets import load_dataset
    import numpy as np
    from transformers import(
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    ctx = current_context()

    working_dir = Path(ctx.working_directory)
    train_dir = working_dir / "models"

    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        cache_dir=model_cache_dir,
    )

    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Use a small subset such that finetuning completes
    small_train_dataset = (
        dataset["train"].shuffle(seed=42).select(range(500)).map(tokenizer_function)
    )
    small_eval_dataset = (
        dataset["test"].shuffle(seed=42).select(range(100)).map(tokenizer_function)
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": np.mean(predictions == labels)}
    
    training_args = TrainingArguments(
        output_dir=train_dir,
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return model

# ---------------------------
# Evaluate model
# ---------------------------
@task(
    container_image=container_image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="12Gi", gpu="1"),
)
def evaluate_model(
    model: BertForSequenceClassification,
    dataset_cache_dir: FlyteDirectory,
    model_cache_dir: FlyteDirectory,
) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np
    import torch

    # Load the test dataset and tokenizer
    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=model_cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Use a small subset (200 examples) for evaluation
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(200)).map(tokenize_function)

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

    # Initialize Trainer for evaluation
    training_args = TrainingArguments(
        output_dir=".", 
        per_device_eval_batch_size=16, 
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    eval_results = trainer.evaluate()

    print(f"Evaluation results on 100 examples: {eval_results}")

    return eval_results

