---

We developed this project as part of the **WIDS 5.0 Final Project**, where we implemented a GPT-style Transformer model entirely from scratch using **PyTorch**. The primary objective was to gain a deeper, hands-on understanding of how the GPT and ChatGPT architectures function internally by manually building every major component, without relying on any prebuilt Transformer libraries.

The model operates at the **character level** and is trained on a large plain-text dataset. It predicts the next character in a given sequence using a masked self-attention mechanism, enabling text generation in an autoregressive, one-character-at-a-time manner.

### Key Features of the Project:

* Decoder-only Transformer architecture
* Autoregressive character-level language model
* Manual implementation of the self-attention mechanism using **Query, Key, and Value (QKV)**
* Causal masking to prevent the model from accessing future tokens during generation
* Multi-head self-attention for learning diverse contextual relationships
* Feed-forward neural networks with non-linear activation functions
* Residual connections and Layer Normalization applied to every sub-layer for stable training
* Positional embeddings to encode sequence order information
* A single Python file (`gpt.py`) supporting both end-to-end training and text generation

### Model Architecture Overview

The architecture consists of multiple Transformer blocks stacked sequentially, following the standard GPT design. Each input character is first mapped to a dense vector using a token embedding table. Positional embeddings are then added to preserve the order of characters in the sequence.

These combined embeddings are passed through several Transformer blocks. Each block contains masked multi-head self-attention followed by a feed-forward neural network. Residual connections and layer normalization are applied around both components to ensure smoother gradient flow and more effective learning.

After passing through all Transformer blocks, the final hidden representations are normalized and projected through a linear layer to produce logits over the vocabulary. These logits represent the probability distribution for predicting the next character.

### Dataset and Training Setup

The training dataset is a plain-text file (~1.1 MB) containing samples from the public-domain works of **William Shakespeare**. The dataset was split into training and validation sets using a 90:10 ratio to evaluate the model’s generalization performance.

The model was trained using the **PyTorch** framework with the **AdamW optimizer** and **Cross Entropy Loss** as the objective function.

Training configuration:

* Training iterations: ~4000
* Batch size: 32
* Context length (block size): 64
* Learning rate: 3e-4
* Dropout rate: 0.2

Training and validation losses were periodically logged to monitor learning progress and identify potential overfitting.

### Results and Observations

* Final training loss: ~1.89
* Final validation loss: ~1.97

The validation loss met the project’s target benchmarks, indicating that the model successfully learned meaningful linguistic patterns. The generated outputs include recognizable English words, sentence-like structures, and dialogue patterns consistent with a character-level Transformer trained from random initialization.

### Text Generation Behavior

The model generates text using standard autoregressive sampling, selecting the next character based on predicted probability distributions. By adjusting the sampling temperature, a balance can be achieved between coherence and randomness. While grammatical correctness is not always perfect, the generated text consistently resembles valid English, including appropriate capitalization and word-like formations.

### Repository Contents

* `gpt.py` – Full implementation of the decoder-only character-level GPT model
* `data.txt` – Training dataset
* `README.md` – Project documentation
* Screenshots – Training logs and sample generated outputs

### What I Learned

Through this project, I gained:

* A strong conceptual and practical understanding of Transformer architectures and self-attention mechanisms
* Insight into how autoregressive language models function internally
* An appreciation for the impact and limits of learning rate selection
* Experience building deep learning models completely from the ground up
* A foundation for applying this knowledge when developing API-driven or AI-powered applications

---


