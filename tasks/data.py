"""
This module contains tasks for downloading the dataset and visualizing the data.
"""

from pathlib import Path

from datasets import Dataset
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from typing_extensions import Annotated
from union import Artifact, Deck, Resources, current_context, task

from containers import container_image

# Define Artifact Specifications
RawImdbDataset = Artifact(name="raw_imdb_dataset")
TrainImdbDataset = Artifact(name="train_imdb_dataset")
ValImdbDataset = Artifact(name="val_imdb_dataset")
TestImdbDataset = Artifact(name="test_imdb_dataset")


# ---------------------------
# download dataset
# ---------------------------
@task(
    container_image=container_image,
    cache=True,
    cache_version="1",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_dataset() -> tuple[
    Annotated[FlyteFile, TrainImdbDataset],
    Annotated[FlyteFile, ValImdbDataset],
    Annotated[FlyteFile, TestImdbDataset],
]:

    import pandas as pd
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    # Load IMDB dataset
    dataset = load_dataset("imdb")
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Split training set into train and validation sets
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42
    )

    working_dir = Path(current_context().working_directory)
    data_dir = working_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets as CSV files
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return (
        TrainImdbDataset.create_from(train_path),
        ValImdbDataset.create_from(val_path),
        TestImdbDataset.create_from(test_path),
    )


# ---------------------------
# visualize data
# ---------------------------
@task(
    container_image=container_image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="2Gi"),
)
def visualize_data(
    train_dataset: FlyteFile, val_dataset: FlyteFile, test_dataset: FlyteFile
):
    import base64
    from textwrap import dedent

    import matplotlib.pyplot as plt
    import pandas as pd

    ctx = current_context()

    # Load datasets from CSV files
    train_df = pd.read_csv(train_dataset.download())
    val_df = pd.read_csv(val_dataset.download())
    test_df = pd.read_csv(test_dataset.download())

    # Create the deck for visualization
    deck = Deck("Dataset Analysis")

    # Sample reviews from the datasets
    train_positive_review = train_df[train_df["label"] == 1].iloc[0]["text"]
    train_negative_review = train_df[train_df["label"] == 0].iloc[0]["text"]
    val_positive_review = val_df[val_df["label"] == 1].iloc[0]["text"]
    val_negative_review = val_df[val_df["label"] == 0].iloc[0]["text"]
    test_positive_review = test_df[test_df["label"] == 1].iloc[0]["text"]
    test_negative_review = test_df[test_df["label"] == 0].iloc[0]["text"]

    # Visualization helper
    def plot_label_distribution(df, title, color, output_path):
        plt.figure(figsize=(10, 5))
        df["label"].value_counts().plot(kind="bar", color=color)
        plt.title(title)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Plot label distributions
    plot_label_distribution(
        train_df,
        "Train Data Label Distribution",
        "skyblue",
        "/tmp/train_label_distribution.png",
    )
    plot_label_distribution(
        val_df,
        "Validation Data Label Distribution",
        "orange",
        "/tmp/val_label_distribution.png",
    )
    plot_label_distribution(
        test_df,
        "Test Data Label Distribution",
        "lightgreen",
        "/tmp/test_label_distribution.png",
    )

    # Convert images to base64 for embedding
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    train_image_base64 = image_to_base64("/tmp/train_label_distribution.png")
    val_image_base64 = image_to_base64("/tmp/val_label_distribution.png")
    test_image_base64 = image_to_base64("/tmp/test_label_distribution.png")

    # Create HTML report
    html_report = dedent(
        f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">Dataset Analysis</h2>

        <h3 style="color: #2980B9;">Training Data Summary</h3>
        <img src="data:image/png;base64,{train_image_base64}" alt="Train Data Label Distribution" width="600">
        Shape: {train_df.shape} <br>
        Label Distribution: {train_df['label'].value_counts()} <br>
        <p><strong>Positive Review:</strong> {train_positive_review}</p>
        <p><strong>Negative Review:</strong> {train_negative_review}</p>

        <h3 style="color: #2980B9;">Validation Data Summary</h3>
        <img src="data:image/png;base64,{val_image_base64}" alt="Validation Data Label Distribution" width="600">
        Shape: {val_df.shape} <br>
        Label Distribution: {val_df['label'].value_counts()} <br>
        <p><strong>Positive Review:</strong> {val_positive_review}</p>
        <p><strong>Negative Review:</strong> {val_negative_review}</p>

        <h3 style="color: #2980B9;">Test Data Summary</h3>
        <img src="data:image/png;base64,{test_image_base64}" alt="Test Data Label Distribution" width="600">
        Shape: {test_df.shape} <br>
        Label Distribution: {test_df['label'].value_counts()} <br>
        <p><strong>Positive Review:</strong> {test_positive_review}</p>
        <p><strong>Negative Review:</strong> {test_negative_review}</p>
    </div>
    """
    )

    # Append HTML content to the deck
    deck.append(html_report)

    # Insert the deck into the context
    ctx.decks.insert(0, deck)
