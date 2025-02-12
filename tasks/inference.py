from flytekit import task, Resources, workflow
from flytekit.types.directory import FlyteDirectory
from containers import container_image, actor
import union


# ---------------------------
# Batch inference with Regular Containers
# ---------------------------
@task(
    container_image=container_image,
    requests=Resources(cpu="2", mem="2Gi", gpu="1"),
    retries=2,
)
def predict_batch_sentiment(trained_model_dir: FlyteDirectory, texts: list[str]) -> list[dict]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    # Download and load the model and tokenizer
    model_dir = trained_model_dir.download()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Initialize the pipeline for sentiment analysis
    nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Perform batch prediction
    predictions = nlp_pipeline(texts, batch_size=8)

    return predictions

# ---------------------------
# Batch inference with Union Actors
# ---------------------------

# @union.actor_cache
# def load_model(trained_model_dir: FlyteDirectory) -> KNeighborsClassifier:
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

#     # actor caching is useful for large models
#     model_dir = trained_model_dir.download()
#     model = AutoModelForSequenceClassification.from_pretrained(model_dir)
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)

#     # Initialize the pipeline for sentiment analysis
#     nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
#     return nlp_pipeline

# @actor.task
# def actor_model_predict(
#    trained_model_dir: FlyteDirectory, texts: list[str]
# ) -> List[int]:
#     nlp_pipeline = load_model(trained_model_dir)

#     predictions = nlp_pipeline(texts, batch_size=8)

#     return predictions.tolist()