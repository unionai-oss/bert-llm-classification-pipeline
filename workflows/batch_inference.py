from flytekit import task, Resources, workflow, Artifact
from flytekit.types.directory import FlyteDirectory
from containers import container_image
from tasks.inference import predict_batch_sentiment, actor_model_predict

FineTunedImdbModel = Artifact(name="fine_tuned_Imdb_model")

# ---------------------------
# Batch inference with regular containers
# ---------------------------
@workflow
def batch_inference_workflow(texts: list[str],
                              trained_model_dir: FlyteDirectory = FineTunedImdbModel.query()
                              ) -> list[dict]:
    # predict_batch_sentiment(trained_model_dir=trained_model_dir, texts=texts)
    pred = predict_batch_sentiment(trained_model_dir=trained_model_dir, texts=texts)
    return pred


# ---------------------------
# Faster Batch inference with Actors
# ---------------------------
@workflow
def actor_batch_inference_workflow(texts: list[str],
                              trained_model_dir: FlyteDirectory = FineTunedImdbModel.query()
                              ) -> list[dict]:
    pred = actor_model_predict(trained_model_dir=trained_model_dir, texts=texts)
    return pred