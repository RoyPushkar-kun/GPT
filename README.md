Bigram Language Model
This repository contains a minimal character-level language model implemented in PyTorch, inspired by the early foundations of GPT-style models.
The goal of this project is learning, not performance — it demonstrates how a neural network can learn to predict the next character in a sequence using a bigram language model.

This Project Does
Trains a character-level language model on the Tiny Shakespeare dataset
Learns next-character prediction using only the previous character
Generates new text character-by-character after training
Introduces key concepts behind GPT-style models:
Tokenization
Embeddings
Cross-entropy loss
Autoregressive generation

Project Structure
.
├── input.txt        # Training data (Tiny Shakespeare)
├── model.py / main.py
└── README.md

