# sLLMLab

> sLLMLab: A minimal ChatGPT-style LLM built from scratch, inspired by GPT-1/2/3 and InstructGPT. Designed for exploration, education, and rapid experimentation, sLLMLab successfully replicates core InstructGPT methods and aims to progressively introduce advanced, state-of-the-art techniques to efficiently train intelligent LLMs on limited GPU resources.
---

## Introduction

**sLLMLab** is a lightweight yet complete training lab for language model development, including all major stages:

- Pretraining
- Supervised Finetuning (SFT)
- Reward Modeling using automatic labeling
- Reinforcement Learning with Human Feedback (RLHF)

Inspired by GPT-1/2/3 and InstructGPT, sLLMLab emphasizes modularity, clarity, and ease of use, supporting ongoing experimentation and exploration.

### Highlights

- Functional PPO-based RLHF implementation
- Clear separation of pipeline stages with prep and train scripts
- Clean and well-structured YAML configuration
- Automatic annotation via external LLMs (e.g., Gemma3)
- Minimal functional version tested on NVIDIA RTX 3060
- Planned extensions: Mixture-of-Experts (MoE), FP8, GRPO
---

## Usage

### Data Preparation

```bash
python pretraining_prepare_data.py
python sft_prepare_data.py
python reward_model_prepare_data.py
python rlhf_prepare_data.py
```

### Training Pipeline

```bash
python pretraining.py --train
python sft.py --train
python reward_model.py --train

# Note: Ollama or other LLM server must be running for reward model training
# Future updates may remove this dependency

python rlhf.py --train
```

### Inference

```bash
python inference.py "Hello"
```

---

## Environment

This repository provides a Conda environment (environment.yml). Please follow these steps:

```bash
conda env create -f environment.yml
conda activate sLLMLab-dev
```

-Tested OS: Ubuntu 24.04.2 LTS

---

## Acknowledgements

- Pretraining data: [C4 dataset](https://www.tensorflow.org/datasets/catalog/c4)
- SFT dataset: [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) (licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/))
- Transformer backbone: Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)
- PPO logic: Adapted in part from [Phil Tabor's PPO implementation](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch)

---

## License

This project is licensed under the MIT License.


