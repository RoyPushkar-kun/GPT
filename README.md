---
# Mini GPT – Bigram Language Model (PyTorch)

This repository contains a **minimal character-level language model** implemented in **PyTorch**, inspired by the early foundations of GPT-style models.

The goal of this project is **learning**, not performance — it demonstrates how a neural network can learn to predict the next character in a sequence using a **bigram language model**.

---

##  What This Project Does

* Trains a **character-level language model** on the Tiny Shakespeare dataset
* Learns **next-character prediction** using only the previous character
* Generates new text character-by-character after training
* Introduces key concepts behind GPT-style models:

  * Tokenization
  * Embeddings
  * Cross-entropy loss
  * Autoregressive generation

>  This is **not a Transformer yet** — it’s a **stepping stone** toward understanding GPT and attention-based models.

---

##  Model Overview

### Bigram Language Model

Each character directly predicts the **next character** using an embedding lookup table.

Mathematically:

```
P(xₜ₊₁ | xₜ)
```

There is **no attention**, **no recurrence**, and **no context beyond one character**.

---

##  Project Structure

```
.
├── input.txt        # Training data (Tiny Shakespeare)
├── model.py / main.py
└── README.md
```

---

##  Dataset

* **Tiny Shakespeare**
* Character-level modeling
* Automatically builds vocabulary from the dataset

---

##  Hyperparameters

| Parameter      | Value      |
| -------------- | ---------- |
| Batch Size     | 32         |
| Block Size     | 8          |
| Learning Rate  | 1e-2       |
| Max Iterations | 3000       |
| Eval Interval  | 300        |
| Optimizer      | AdamW      |
| Device         | CPU / CUDA |

---

##  Training Process

* Random batches of character sequences are sampled
* Model predicts the next character for each position
* Loss is calculated using **CrossEntropyLoss**
* Evaluated periodically on train and validation sets

---

##  Text Generation

After training, the model generates text by:

1. Starting with an initial token
2. Predicting probabilities for the next character
3. Sampling from the probability distribution
4. Appending the sampled character
5. Repeating autoregressively

Example:

```python
context = torch.zeros((1, 1), dtype=torch.long)
model.generate(context, max_new_tokens=500)
```

---

