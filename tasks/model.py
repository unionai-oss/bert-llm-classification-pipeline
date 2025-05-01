"""
This file contains the tasks that are used to download, train and evaluate the model.
"""

from pathlib import Path
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from typing_extensions import Annotated
from union import Artifact, Deck, Resources, current_context, task
from containers import container_image

# Define Artifact Specifications
FineTunedImdbModel = Artifact(name="fine_tuned_Imdb_model")

# ---------------------------
# download model
# ---------------------------
@task(
    container_image=container_image,
    cache=True,
    cache_version="1",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_model(model_name: str) -> FlyteDirectory:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    working_dir = Path(current_context().working_directory)
    saved_model_dir = working_dir / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)

    return FlyteDirectory(saved_model_dir)


# ---------------------------
# full/lora/qlora fine-tune model
# ---------------------------
@task(
    container_image=container_image,
    requests=Resources(cpu="4", mem="12Gi", gpu="1"),
)
def train_model(
    model_dir: FlyteDirectory,
    train_dataset: FlyteFile,
    val_dataset: FlyteFile,
    epochs: int = 3,
    tuning_method: str = "full",  # options: "full", "lora", "qlora"
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> Annotated[FlyteDirectory, FineTunedImdbModel]:
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    local_model_dir = model_dir.download()
    train_df = pd.read_csv(train_dataset.download()).sample(n=500, random_state=42)
    val_df = pd.read_csv(val_dataset.download()).sample(n=100, random_state=42)

    train_dataset_hf = Dataset.from_pandas(train_df)
    val_dataset_hf = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

    def tokenizer_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_train = train_dataset_hf.map(tokenizer_function)
    tokenized_val = val_dataset_hf.map(tokenizer_function)

    if tuning_method == "qlora":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["classifier", "pre_classifier"],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            local_model_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            # device_map="auto", # use this for most models if implemented
        )

    else:
        model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)

    if tuning_method in {"lora", "qlora"}:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "k_lin", "v_lin"], # query, Key, Value linear layers in this model
        )
 
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    trainer.train()

    # Merge LoRA weights into base model
    if tuning_method in {"lora", "qlora"}:
        model = model.merge_and_unload()

    output_dir = Path(current_context().working_directory) / "trained_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    #TODO: Save traning type (lora, qlora, full) as artifacts
    return FineTunedImdbModel.create_from(output_dir)



# ---------------------------
# evaluate model
# ---------------------------
@task(
    container_image=container_image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="12Gi", gpu="1"),
)
def evaluate_model(trained_model_dir: FlyteDirectory, test_dataset: FlyteFile) -> dict:
    import numpy as np
    import pandas as pd
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import seaborn as sns
    import matplotlib.pyplot as plt
    import base64
    from union import current_context
    from union import Deck
    from textwrap import dedent

    # Download model locally
    local_model_dir = trained_model_dir.download()
    ctx = current_context()

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_dir,
        torch_dtype="auto",
        load_in_4bit=False,  # Important: for evaluation, avoid loading in quantized 4-bit unless you really want to
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

    # Load and prepare the test dataset
    test_df = pd.read_csv(test_dataset.download()).sample(n=100, random_state=42)

    # Use a pipeline for evaluation (bypasses Trainer and works for quantized models)
    nlp_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        # device=0 if torch.cuda.is_available() else -1,  # auto-select device
        truncation=True,
        padding=True,
    )

    # Perform batch inference
    predictions = nlp_pipeline(test_df["text"].tolist(), batch_size=8)

    # Extract predicted labels
    pred_labels = [int(p["label"].split("_")[-1]) if "label" in p else 0 for p in predictions]
    true_labels = test_df["label"].tolist()

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels, average="weighted"),
        "precision": precision_score(true_labels, pred_labels, average="weighted"),
        "recall": recall_score(true_labels, pred_labels, average="weighted"),
        # "conf_matrix": confusion_matrix(true_labels, pred_labels)
    }

    # create visualization deck
    deck = Deck("Model Evaluation")

    # Generate Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_path = f"/tmp/confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(true_labels)), yticklabels=sorted(set(true_labels)))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    
    # # Generate ROC Curve
    # if len(set(true_labels)) == 2:  # Only for binary classification
    #     fpr, tpr, _ = roc_curve(true_labels, probs[:, 1])
    #     roc_auc = auc(fpr, tpr)
    #     roc_path = f"tmp/roc_curve.png"
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
    #     plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    #     plt.savefig(roc_path)
    #     plt.close()
    # else:
    #     roc_path = None
    
    # Convert images to base64 for embedding
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
        
    cm_image_base64 = image_to_base64(cm_path)
    # roc_image_base64 = image_to_base64(roc_path) if roc_path else None

    # Create HTML report
    html_report = dedent(
        f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">Model Evaluation</h2>

        <h3 style="color: #2980B9;">Confusion Matrix</h3>
        <img src="data:image/png;base64,{cm_image_base64}" alt="Confusion Matrix" width="600">
        <h3 style="color: #2980B9;">Model Metrics</h3>
        <pre>{metrics}</pre>
        
    </div>
        """)

     # Append HTML content to the deck
    deck.append(html_report)
    # Insert the deck into the context
    ctx.decks.insert(0, deck)


    return metrics
