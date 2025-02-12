'''
This module contains tasks for downloading the dataset and visualizing the data.
'''

from union import task, Resources, current_context, Deck, Artifact
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from pathlib import Path
from datasets import Dataset
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
    cache_version="0.001",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_dataset() -> tuple[Dataset, Dataset, Dataset]:
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Load IMDB dataset
    dataset = load_dataset("imdb")
    df = dataset['train'].to_pandas()

    # Stratified split for validation
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    test_df = dataset['test'].to_pandas()

    # Convert back to Hugging Face datasets and return directly
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset


# ---------------------------
# visualize data
# ---------------------------
@task(
    container_image=container_image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="2Gi"),
)
def visualize_data(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
    import matplotlib.pyplot as plt
    import pandas as pd
    import base64
    from textwrap import dedent

    ctx = current_context()

    # Convert datasets to DataFrames for visualization
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)
    test_df = pd.DataFrame(test_dataset)

    # Create the deck for visualization
    deck = Deck("Dataset Analysis")

    # Sample reviews from training, validation, and test datasets
    train_positive_review = train_df[train_df['label'] == 1].iloc[0]['text']
    train_negative_review = train_df[train_df['label'] == 0].iloc[0]['text']
    val_positive_review = val_df[val_df['label'] == 1].iloc[0]['text']
    val_negative_review = val_df[val_df['label'] == 0].iloc[0]['text']
    test_positive_review = test_df[test_df['label'] == 1].iloc[0]['text']
    test_negative_review = test_df[test_df['label'] == 0].iloc[0]['text']

    # Visualization helper
    def plot_label_distribution(df, title, color, output_path):
        plt.figure(figsize=(10, 5))
        df['label'].value_counts().plot(kind='bar', color=color)
        plt.title(title)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Plot label distributions
    plot_label_distribution(train_df, 'Train Data Label Distribution', 'skyblue', '/tmp/train_label_distribution.png')
    plot_label_distribution(val_df, 'Validation Data Label Distribution', 'orange', '/tmp/val_label_distribution.png')
    plot_label_distribution(test_df, 'Test Data Label Distribution', 'lightgreen', '/tmp/test_label_distribution.png')

    # Convert images to base64 for embedding
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    train_image_base64 = image_to_base64('/tmp/train_label_distribution.png')
    val_image_base64 = image_to_base64('/tmp/val_label_distribution.png')
    test_image_base64 = image_to_base64('/tmp/test_label_distribution.png')

    # Create HTML report
    html_report = dedent(f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">Dataset Analysis</h2>

        <h3 style="color: #2980B9;">Training Data Summary</h3>
        Shape: {train_df.shape} <br>
        Label Distribution: {train_df['label'].value_counts()} <br>
        <p><strong>Positive Review:</strong> {train_positive_review}</p>
        <p><strong>Negative Review:</strong> {train_negative_review}</p>
        <img src="data:image/png;base64,{train_image_base64}" alt="Train Data Label Distribution" width="600">

        <h3 style="color: #2980B9;">Validation Data Summary</h3>
        Shape: {val_df.shape} <br>
        Label Distribution: {val_df['label'].value_counts()} <br>
        <p><strong>Positive Review:</strong> {val_positive_review}</p>
        <p><strong>Negative Review:</strong> {val_negative_review}</p>
        <img src="data:image/png;base64,{val_image_base64}" alt="Validation Data Label Distribution" width="600">

        <h3 style="color: #2980B9;">Test Data Summary</h3>
        Shape: {test_df.shape} <br>
        Label Distribution: {test_df['label'].value_counts()} <br>
        <p><strong>Positive Review:</strong> {test_positive_review}</p>
        <p><strong>Negative Review:</strong> {test_negative_review}</p>
        <img src="data:image/png;base64,{test_image_base64}" alt="Test Data Label Distribution" width="600">
    </div>
    """)

    # Append HTML content to the deck
    deck.append(html_report)

    # Insert the deck into the context
    ctx.decks.insert(0, deck)
