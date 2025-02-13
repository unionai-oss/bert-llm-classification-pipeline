# BERT-LLM Classification Fine-Tuning and Serving Pipeline

This repository contains a pipeline for fine-tuning a BERT-LLM model on a classification task and serving the model using the Union AI workflow and inference platform. 

## Project Setup
The quickest way to setup and the run the tutorial notebook is often using a hosted notebook, like Google colab.

<a target="_blank" href="https://colab.research.google.com/github/unionai-oss/bert-llm-classification-pipeline/blob/main/tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Sign up for Union account
Serverless is the easiest way to get started with Union. You can sign up for a free account and $30 of credit at [Union Serverless](https://app.union.ai/signup). BYOC (Bring Your Own Cloud) is also available for more features and advanced users. [Schedule a demo](https://www.union.ai/contact) to learn more about Union BYOC.

- Union Serverless Sign-up: 
- Union BYOC: https://docs.union.ai/byoc/user-guide/union-overview#union-byoc

Read more in the overview of Union Serverless and Union BYOC.

### Authenticate to Union from CLI
After you have signed up for Union, you can authenticate to Union from the CLI.

If on Union Serverless
`union create login --serverless --auth device-flow`

If on Union BYOC (Bring Your Own Cloud)
`union create login --host <union-host-url>`

Now your environment is setup to run the project on remotely Union.

- [Autneticatoin docs](https://docs.union.ai/serverless/api-reference/union-cli#configure-the-union-cli)