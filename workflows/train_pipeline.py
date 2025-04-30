"""
This file contains the train_pipeline workflow that orchestrates the
training pipeline for BERT classification models
"""

from union import workflow

from tasks.data import download_dataset, visualize_data
from tasks.inference import predict_batch_sentiment
from tasks.model import download_model, evaluate_model, train_model


# ---------------------------
# train pipeline
# ---------------------------
@workflow
def train_pipeline(
    tuning_method: str = "lora",  # options: "full", "lora", "qlora"
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    extra_test_text: list[str] = [
        "This is a great movie!",
        "This is a bad movie!",
    ],
) -> None:

    train_dataset, val_dataset, test_dataset = download_dataset()
    saved_model_dir = download_model(model_name=model_name)

    visualize_data(
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset
    )

    trained_model_dir = train_model(
        tuning_method=tuning_method,
        model_dir=saved_model_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
    )

    evaluate_model(trained_model_dir=trained_model_dir, test_dataset=test_dataset)

    # Perform batch inference
    predict_batch_sentiment(trained_model_dir=trained_model_dir, texts=extra_test_text)

# Run model training pipeline:
# union run --remote workflows/train_pipeline.py train_pipeline --epochs 3 --tuning_method full 