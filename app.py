"""A Union app that uses hugging face and Streamlit"""

import os

from union import Artifact, ImageSpec, Resources
from union.app import App, Input, ScalingMetric
from datetime import timedelta


# Define the artifact that holds the BERT model.
FineTunedImdbModel = Artifact(name="fine_tuned_Imdb_model")

# Define the container image including the required packages.
image_spec = ImageSpec(
    name="union-serve-bert-sentiment-analysis",
    packages=[
        "transformers==4.48.3",
        "union-runtime>=0.1.11",
        "accelerate==1.3.0",
        "streamlit==1.43.2",  # For the UI
    ],
    registry=os.getenv("REGISTRY"),
)

# Create the Union Serving App.
streamlit_app = App(
    name="bert-sentiment-analysis",
    inputs=[
        Input(
            name="bert_model",
            value=FineTunedImdbModel.query(),
            download=True,  # The model artifact is downloaded when the container starts.
        )
    ],
    container_image=image_spec,
    limits=Resources(cpu="1", mem="4Gi", gpu="1"),
    port=8082,
    include=["./main.py"],  # Include your Streamlit code.
    # args=["streamlit", "run", "main.py", "--server.port", "8082"],
    args="streamlit run main.py --server.port 8082",
    min_replicas=0,
    max_replicas=2,
    scaledown_after=timedelta(minutes=5),
    scaling_metric=ScalingMetric.Concurrency(2),
    # requires_auth=False # Uncomment to make app public.
)

# union deploy apps app.py bert-sentiment-analysis
