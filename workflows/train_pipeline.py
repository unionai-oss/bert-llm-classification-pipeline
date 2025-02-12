'''
This file contains the train_pipeline workflow that orchestrates the
training pipeline for BERT classification models
'''

from union import workflow
from tasks.data import download_dataset, visualize_data
from tasks.model import download_model, train_model, evaluate_model

# ---------------------------
# train pipeline
# ---------------------------
@workflow
def train_pipeline(model_name: str = "bert-base-uncased", epochs: int = 3) -> None:
    train_dataset, val_dataset, test_dataset = download_dataset()

    saved_model_dir = download_model(model_name=model_name)
    
    visualize_data(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

    trained_model_dir = train_model(
        model_dir=saved_model_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs
    )

    evaluate_model(
        trained_model_dir=trained_model_dir,
        test_dataset=test_dataset
    )

# Run model training pipeline:
#!union run --remote workflows/train_pipeline.py train_pipeline
