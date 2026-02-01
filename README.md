
---

# WiDS Transformer

This repository documents the development of a **character-level language model** built as part of the **WIDS (Winter in Data Science) 2026 Final Project**. The project traces the evolution from a simple **Bigram baseline** to a fully functional **Decoder-Only Transformer (GPT-style architecture)** implemented **entirely from scratch using PyTorch**, without relying on any existing Transformer libraries.

The primary motivation behind this work was to gain a deep, practical understanding of how modern large language models like GPT and ChatGPT function internally—by manually implementing every major architectural component.

---

## 1. Project Overview

The core objective of this project was to transition from *local character prediction* to *context-aware language modeling*. Early versions of the model relied only on immediate character transitions, while the final implementation leverages **masked self-attention** to capture long-range dependencies and sentence-level structure.

The final model is an **autoregressive, decoder-only Transformer** trained at the character level to predict the next character in a sequence, enabling text generation one character at a time.

---

### Phase 1: Bigram Baseline

* **Concept:** Predicts the next character using only the current character.
* **Limitation:** No memory or understanding of context.
* **Outcome:** Generates mostly gibberish, but still learns basic statistical patterns such as capitalization after newlines and common character combinations.
* **Validation Loss:** ~2.53

This phase served as a baseline and motivation for introducing contextual modeling.

---

### Phase 2: GPT-Style Transformer

* **Concept:** A decoder-only Transformer architecture with positional embeddings and masked self-attention.
* **Goal:** Enable contextual understanding and generate coherent English-like text.
* **Target Validation Loss:** 1.5 – 2.0
* **Final Result:** Validation loss comfortably within the target range, with clear improvements in spelling, sentence structure, and dialogue flow.

---

## 2. Model Architecture (`gpt.py`)

The final implementation is a **from-scratch GPT-style Transformer**, built as a single file that supports both training and text generation.

### Core Components

* **Token Embeddings:** Convert characters into dense vector representations.
* **Positional Embeddings:** Encode sequence order information.
* **Head:** A single masked self-attention head computing Query, Key, and Value matrices.
* **MultiHeadAttention:** Multiple attention heads operating in parallel to capture diverse linguistic relationships.
* **FeedForward Network:** Two-layer neural network with non-linear activation (ReLU).
* **Block:** A complete Transformer block combining attention, feed-forward layers, residual connections, and layer normalization.
* **Causal Masking:** Prevents access to future tokens during training and generation.

Residual connections and layer normalization are applied around every sub-layer to ensure stable training and smoother gradient flow.

---

## 3. Training Setup & Implementation Details

* **Dataset:** `tinyshakespeare` (≈1.1M characters, public domain).
* **Tokenizer:** Character-level integer encoding using `stoi` and `itos`.
* **Train / Validation Split:** 90% / 10%
* **Optimizer:** AdamW
* **Loss Function:** Cross Entropy Loss

### Hyperparameters

* **Batch Size:** 64
* **Context Length (Block Size):** 256
* **Embedding Dimension:** 384
* **Attention Heads:** 6
* **Transformer Layers:** 6
* **Training Iterations:** ~5,000
* **Learning Rate:** 3e-4
* **Dropout:** 0.2

Training and validation losses are logged periodically to monitor convergence and detect overfitting.

---

## 4. Performance & Results

The model was trained for approximately **5,000 iterations**, moving well beyond memorization and into meaningful pattern learning.

| Metric                     | Value            |
| -------------------------- | ---------------- |
| **Initial Loss**           | ~4.28            |
| **Final Training Loss**    | ~1.89            |
| **Final Validation Loss**  | **~1.49 – 1.97** |
| **Target Validation Loss** | 1.5 – 2.0        |

The validation loss met and exceeded project expectations, demonstrating that the model successfully learned meaningful linguistic structure despite being trained entirely from random initialization.

---

### Sample Generated Output

> But with prison: I will stead with you.
>
> **ISABELLA:**
> Carress, all do; and I'll say your honour self good:
> Then I'll regn your highness and
> Compell'd by my sweet gates that you may:
> Valiant make how I heard of you.
>
> **ANGELO:**
> Nay, sir, I say!
>
> **ISABELLA:**
> I am sweet men sister as you steed.

While not grammatically perfect, the output consistently resembles structured English with correct capitalization, dialogue formatting, and Shakespearean tone—expected behavior for a character-level Transformer.

---

## 5. How to Run

1. **Environment:**
   Recommended to use **Google Colab** with a **T4 GPU** for faster training.

2. **Install Dependencies:**

   ```bash
   pip install torch
   ```

3. **Run the Model:**

   ```bash
   python gpt.py
   ```

   The script automatically downloads the training dataset if it is not present.

---

## 6. Repository Contents

* `gpt.py` – Complete decoder-only GPT-style Transformer (training + generation)
* `data.txt / input.txt` – Shakespeare dataset
* `README.md` – Project documentation
* Screenshots – Training logs and generated text outputs

---

## 7. What I Learned

Through this project, I gained:

* A strong conceptual and practical understanding of **Transformer architectures**
* Hands-on experience implementing **self-attention (QKV)** from scratch
* Insight into **autoregressive language modeling**
* Understanding of the effects and limits of **learning rate tuning**
* Experience building deep learning models **from the ground up**
* A solid foundation for developing **API-driven and LLM-based applications**

---

## 8. Acknowledgments

* **Andrej Karpathy** – *“Let’s Build GPT from Scratch”*
* **Harvard NLP** – *The Annotated Transformer*
* **Jay Alammar** – *The Illustrated Transformer*

---


