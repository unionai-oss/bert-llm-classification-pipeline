'''
This file contains the train_pipeline workflow that orchestrates the
training pipeline for BERT classification models
'''

from union import workflow
from tasks.data import download_dataset, visualize_data
from tasks.model import download_model, train_model

# ---------------------------
# train pipeline
# ---------------------------
@workflow
def train_pipeline(model_name: str ="bert-base-uncased",
                    epochs: int = 3) -> None:
    dataset_cache_dir = download_dataset()
    model_cache_dir = download_model(model_name)
    visualize_data(dataset_cache_dir)
    train_model(model_name, dataset_cache_dir, model_cache_dir, epochs)
    # evaluate_model(model_name, dataset_cache_dir, model_cache_dir)
    # predict(model_name, dataset_cache_dir, model_cache_dir)

# run model training pipeline:
#!union run --remote workflows/train_pipeline.py train_pipeline
