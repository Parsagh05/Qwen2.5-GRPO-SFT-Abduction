# Abductive Reasoning Alignment: Qwen2.5-3B with SFT + GRPO

This repository contains the code and resources for fine-tuning the **Qwen2.5-3B-Instruct** language model to enhance its **abductive reasoning** capabilities using the **UniADILR** dataset. The project explores whether improving abductive reasoning skills can transfer to deductive reasoning tasks.

## 🚀 Project Overview

The goal of this project was to align a small language model (3B parameters) to generate structured abductive reasoning outputs—specifically hypotheses and their supporting proofs—and to evaluate if this specific training improves general logical reasoning.

### Key Methodology: SFT + GRPO
To achieve this, I implemented a two-stage training pipeline inspired by recent alignment research (e.g., DeepSeek-R1):

1.  **Stage 1: Supervised Fine-Tuning (SFT)** - A short phase to "lock" the model into a strict output format (using XML-like tags `<hypothesis>` and `<proof>`).
2.  **Stage 2: Group Relative Policy Optimization (GRPO)** - The core reinforcement learning phase where the model learns the *logic* of abduction. This stage uses custom reward functions to optimize for both structural compliance and semantic correctness without breaking the format.

## 🛠️ Methodology & Implementation

### Model & Tools
* **Base Model:** `Qwen/Qwen2.5-3B-Instruct` (loaded in 4-bit via Unsloth for memory efficiency).
* **Dataset:** [UniADILR](https://github.com/YuSheng-00/UniADILR) (Abductive reasoning dataset containing Context, Hypothesis, and Proof).
* **Libraries:** `TRL`, `Unsloth`, `Transformers`, `Peft`.

### Reward Functions
The GRPO training was guided by three specific reward functions:
1.  **Structural Reward:** Enforces strict adherence to the `<hypothesis>` and `<proof>` XML tag format.
2.  **Semantic Reward (F1-Score):** Rewards the semantic similarity between the generated hypothesis and the ground truth using token overlap (F1 score). *Note: F1 was chosen over embedding-based similarity (like `all-MiniLM-L6-v2`) to reduce computational overhead and allow for faster training loops.*
3.  **Proof Exact Reward:** Rewards exact matching of the reasoning proof sentences (e.g., "sent1 & sent3"), as abductive proofs in this dataset require precise citation.

## 📊 Results & Evaluation

### Training Performance
The two-stage approach proved highly effective. The initial SFT phase successfully taught the model the required output structure. The subsequent GRPO phase significantly improved the *correctness* of the generated proofs and hypotheses, demonstrating that the model learned to identify the correct supporting facts.

### Transfer Learning (Deductive Reasoning)
To test the hypothesis that abductive training transfers to deductive capabilities, the final model was evaluated on the **MMLU Formal Logic** benchmark.

| Model | Accuracy (MMLU Formal Logic) | Improvement |
| :--- | :---: | :---: |
| **Base Model (Qwen 2.5-3B Raw)** | 42.0% | - |
| **Fine-Tuned Model (SFT + GRPO)** | **54.0%** | **+12.0%** 🔼 |

**Result:** The model achieved a **12% accuracy increase** on a deductive logic task it was not explicitly trained on. This suggests a positive transfer of reasoning capabilities from the abductive domain to the deductive domain.
