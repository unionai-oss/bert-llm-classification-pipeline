# BERT-LLM Classification Fine-Tuning and Serving Pipeline


<a target="_blank" href="https://colab.research.google.com/github/unionai-oss/bert-llm-classification-pipeline/blob/main/tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This repository contains a pipeline for fine-tuning a BERT-LLM model on a classification task and serving the model using the Union AI workflow and inference platform. 

## Project Setup
The quickest way to setup and the run the tutorial notebook is often using a hosted notebook, like Google colab.

<a target="_blank" href="https://colab.research.google.com/github/unionai-oss/bert-llm-classification-pipeline/blob/main/tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Or you can follow the steps below to setup the project locally.

### Sign up for a Union account
Serverless is the easiest way to get started with Union. You can sign up for a free account and $30 of credit at [Union Serverless](https://signup.union.ai/?page=signup). BYOC (Bring Your Own Cloud) is also available for more features and advanced users. [Schedule a demo](https://www.union.ai/contact) to learn more about Union BYOC.

- Union Serverless Sign-up: https://www.union.ai/ (Click login and then Sign up)
- Union BYOC: https://docs.union.ai/byoc/user-guide/union-overview#union-byoc

Read more in the overview of Union Serverless and Union BYOC.

### Clone the repository
```bash
git clone https://github.com/unionai-oss/bert-llm-classification-pipeline
cd bert-llm-classification-pipeline
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Authenticate to Union from CLI
After you have signed up for Union, you can authenticate to Union from the CLI.

If on Union Serverless
`union create login --serverless --auth device-flow`

If on Union BYOC (Bring Your Own Cloud)
`union create login --host <union-host-url>`

Now your environment is setup to run the project on remotely Union.

- [Autneticatoin docs](https://docs.union.ai/serverless/api-reference/union-cli#configure-the-union-cli)

### Run through the tutorial notebook or run the pipeline from CLI.
The tutorial notebook will guide you through the steps to fine-tune a BERT-LLM model on a classification task and serve the model using Union AI.

```bash
jupyter notebook tutorial.ipynb
```

Or you can run the steps for the training pipeline and serving from the CLI.

Train the model:
```bash
# 🌟 Run the bert training pipeline using lora, qlora or full
union run --remote workflows/train_pipeline.py train_pipeline --epochs 3 --tuning_method full 
```

Serve the model:
```bash
union deploy apps app.py bert-sentiment-analysis
```


