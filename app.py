"""A Union app that uses hugging face and Streamlit"""

import os
from union import Artifact, ImageSpec, Resources
from union.app import App, Input

# Define the artifact that holds the iris model.
FineTunedImdbModel = Artifact(name="fine_tuned_Imdb_model")

# Define the container image including the required packages.
image_spec = ImageSpec(
    name="union-serve-bert-sentiment-analysis",
    packages=[
        "transformers==4.48.3",
        "union-runtime>=0.1.10",
        "accelerate==1.3.0",
        "streamlit",  # For the UI
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
    args=["streamlit", "run", "main.py", "--server.port", "8082"],
)

# union deploy apps app.py simple-streamlit-iris
