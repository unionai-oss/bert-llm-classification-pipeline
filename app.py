"""A Union app that uses hugging face and Streamlit"""

import os

from union import Artifact, ImageSpec, Resources
from union.app import App, Input, ScalingMetric
from datetime import timedelta
from flytekit.extras.accelerators import L4, GPUAccelerator


# Define the artifact that holds the BERT model.
FineTunedImdbModel = Artifact(name="fine_tuned_Imdb_model")

# Define the container image including the required packages.
image_spec = ImageSpec(
    name="union-serve-bert-sentiment-analysis",
    packages=[
        "transformers==4.48.3",
        "union-runtime>=0.1.11",
        "accelerate==1.5.2",
        "streamlit==1.43.2",
        "bitsandbytes==0.45.3"
    ],
    pip_extra_index_url=["https://download.pytorch.org/whl/cu118"],  # ✅ enables +cu118 builds
    cuda="11.8",  # ensure GPU + CUDA layer is available
    apt_packages=["gcc", "g++"],  # optional, for packages like bitsandbytes
    builder="union",
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
    limits=Resources(cpu="2", mem="24Gi", gpu="1", ephemeral_storage="20Gi"),
    requests=Resources(cpu="2", mem="24Gi", gpu="1", ephemeral_storage="20Gi"),
    accelerator=L4,
    port=8082,
    include=["./main.py"],  # Include your Streamlit code.
    args=["streamlit", "run", "main.py", "--server.port", "8082"],
    min_replicas=0,
    max_replicas=1,
    scaledown_after=timedelta(minutes=5),
    scaling_metric=ScalingMetric.Concurrency(2),
    # requires_auth=False # Uncomment to make app public.
)

# union deploy apps app.py bert-sentiment-analysis
