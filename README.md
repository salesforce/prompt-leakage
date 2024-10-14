
# Prompt Leakage effect and defense strategies for multi-turn LLM interactions

This repository contains the data and code for the experiments in the [PromptLeakage](https://arxiv.org/abs/2404.16251v3) EMNLP 2024 paper.

<p align="center">
  <img height="400" src="figure-prompt-leakage.jpg">
</p>

Figure:  Our standardized task setup for evaluating LLM vulnerability against multi-turn prompt leakage




## Table of Contents

1. [Abstract](abstract)
2. [Accessing the Data](#accessing-the-data)
3. [Running evaluations](#running-evaluations-with-different-llms)
4. [Citation](#citation)
5. [Get Involved](#get-involved)


## Abstract

Prompt leakage poses a compelling security and privacy threat in LLM applications. Leakage of system prompts may compromise intellectual property, and act as adversarial reconnaissance for an attacker. A systematic evaluation of prompt leakage threats and mitigation strategies is lacking, especially for multi-turn LLM interactions. In this paper, we systematically investigate LLM vulnerabilities against prompt leakage for 10 closed- and open-source LLMs, across four domains. We design a unique threat model which leverages the LLM sycophancy effect and elevates the average attack success rate (ASR) from 17.7% to 86.2% in a multi-turn setting. Our standardized setup further allows dissecting leakage of specific prompt contents such as task instructions and knowledge documents. We measure the mitigation effect of 7 black-box defense strategies, along with finetuning an open-source model to defend against leakage attempts. We present different combination of defenses against our threat model, including a cost analysis. Our study highlights key takeaways for building secure LLM applications and provides directions for research in multi-turn LLM interactions

## Accessing the Data

We publicly release the datasets used for each of the 4 domains - news, legal, finance and medical, in the `data/` directory. The finetuning data used can be found at `safetyfinetuning/prompt-leakage-finetune.csv`


## Running evaluations with different LLMs

### Export API keys for black box LLMs

`export ANTHROPIC_API_KEY=...` 
`export COHERE_API_KEY=...` 
`export GOOGLE_APPLICATION_CREDENTIALS=...` 
`export OPENAI_API_KEY=...` 

### Initialize ollama for open source LLMs

`bash ollama_init.sh`

### Running threat model for both experiment settings 

`python run_inference_no_defenses.py`
`python run_inference_all_defenses.py`


## Citation

```
@article{agarwal2024promptleakageeffectdefense,
      title={Prompt Leakage effect and defense strategies for multi-turn LLM interactions}, 
      author={Divyansh Agarwal and Alexander R. Fabbri and Ben Risher and Philippe Laban and Shafiq Joty and Chien-Sheng Wu},
      year={2024},
      eprint={2404.16251},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2404.16251}, 
}
```

### Get Involved

Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
We welcome PRs!
