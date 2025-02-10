# %% download dataset
# ---------------------------
@task(
    container_image=image,
    cache=True,
    cache_version="v3",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_dataset() -> FlyteDirectory:
    from datasets import load_dataset
   
    working_dir = Path(current_context().working_directory)
    dataset_cache_dir = working_dir / "dataset_cache"

    load_dataset("imdb", cache_dir=dataset_cache_dir)

    return dataset_cache_dir


# %% visualize data
# ---------------------------
@task(
    container_image=image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="2Gi"),
)
def visualize_data(dataset_cache_dir: FlyteDirectory):
    from datasets import load_dataset
    import matplotlib.pyplot as plt
    import pandas as pd
    import base64
    from textwrap import dedent

    ctx = current_context()

    # Load the dataset
    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # Create the deck for visualization
    deck = Deck("Dataset Analysis")

    # Sample one review from each class (positive and negative) from the training and test datasets
    train_positive_review = train_df[train_df['label'] == 1].iloc[0]['text']
    train_negative_review = train_df[train_df['label'] == 0].iloc[0]['text']
    test_positive_review = test_df[test_df['label'] == 1].iloc[0]['text']
    test_negative_review = test_df[test_df['label'] == 0].iloc[0]['text']

    # Visualize label distribution for training data
    plt.figure(figsize=(10, 5))
    train_df['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Train Data Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    train_label_dist_path = "/tmp/train_label_distribution.png"
    plt.savefig(train_label_dist_path)
    plt.close()

    # Visualize label distribution for test data
    plt.figure(figsize=(10, 5))
    test_df['label'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Test Data Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    test_label_dist_path = "/tmp/test_label_distribution.png"
    plt.savefig(test_label_dist_path)
    plt.close()

    # Convert images to base64 and embed in HTML
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    train_image_base64 = image_to_base64(train_label_dist_path)
    test_image_base64 = image_to_base64(test_label_dist_path)

    # HTML report with styled text, tables, and embedded images
    html_report = dedent(f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">Dataset Analysis</h2>
        
        <h3 style="color: #2980B9;">Training Data Summary</h3>
        <p>Below is a summary of the training dataset including the distribution of labels.</p>
        Shape: {train_df.shape} <br>
        Columns: {train_df.columns} <br>
        Label Distribution: {train_df['label'].value_counts()} <br>
        
        <h3 style="color: #2980B9;">Sample Reviews from Training Data</h3>
        <p><strong>Positive Review:</strong> {train_positive_review}</p>
        <p><strong>Negative Review:</strong> {train_negative_review}</p>

        <h3 style="color: #2980B9;">Training Data Label Distribution</h3>
        <p>The following bar chart shows the distribution of labels in the training dataset:</p>
        <img src="data:image/png;base64,{train_image_base64}" alt="Train Data Label Distribution" width="600">

        <h3 style="color: #2980B9;">Test Data Summary</h3>
        <p>Below is a summary of the test dataset including the distribution of labels.</p>
        Shape: {test_df.shape} <br>
        Columns: {test_df.columns} <br>
        Label Distribution: {test_df['label'].value_counts()} <br>
        
        <h3 style="color: #2980B9;">Sample Reviews from Test Data</h3>
        <p><strong>Positive Review:</strong> {test_positive_review}</p>
        <p><strong>Negative Review:</strong> {test_negative_review}</p>

        <h3 style="color: #2980B9;">Test Data Label Distribution</h3>
        <p>The following bar chart shows the distribution of labels in the test dataset:</p>
        <img src="data:image/png;base64,{test_image_base64}" alt="Test Data Label Distribution" width="600">
    </div>
    """)

    # Append HTML content to the deck
    deck.append(html_report)

    # Insert the deck into the context
    ctx.decks.insert(0, deck)
