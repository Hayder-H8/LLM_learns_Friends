
<h1 align='center'>LLM_learns_Friends</h1>

<p align="center">
  <b>LLM_learns_Friends</b> is a from-scratch implementation of a Byte-Pair Encoding (BPE) tokenizer and a Transformer-based language model.  
  It is trained on the full <b>F.R.I.E.N.D.S</b> series transcript with the goal of generating new, plausible dialogue given, for example, an episode title or context prompt.
</p>

---

## 🧠 Project Overview

This project aims to demonstrate the end-to-end process of building a small Large Language Model (LLM) — from tokenization to training and text generation — using a lightweight transformer architecture.

The workflow is divided into several key modules:

| File | Description |
|------|--------------|
| `tokenizer.py` | Implements a Byte-Pair Encoding (BPE) tokenizer, including vocabulary construction, merges, encoding, and decoding. |
| `tokenizer_utils.py` | Utility functions for preprocessing, corpus reading, and token serialization/deserialization. |
| `data.py` | Implements the `batch_generator` class that creates train/validation splits and context-sized token batches. |
| `model.py` | Defines the Transformer-based model (`model_predictor`) with self-attention layers, feed-forward blocks, and embedding layers. |
| `train.py` | Training and evaluation loops, including checkpointing with `torch.save(model.state_dict())`. |
| `main.py` | The CLI entry point, which uses `argparse` to configure tokenizer, model, and training hyperparameters. |
| `Friends_Transcript/` | Directory containing the full F.R.I.E.N.D.S script used as corpus data. |
| `vocabulary.json` | Serialized tokenizer vocabulary and merges. |

---

## ⚙️ Installation & Requirements

You can run this project using Python ≥ 3.10.  
It is recommended to use a dedicated conda or virtual environment.

```bash
# Create and activate environment
conda create -n llm-friends python=3.10 -y
conda activate llm-friends

# Install dependencies
pip install torch numpy regex

## 🚀 Usage

The project is entirely configurable through command-line arguments.  
All constants (context size, merges, layers, etc.) can be changed through `--args`.

### 🏋️ Train or Resume Training

To train the model from scratch or continue training from a checkpoint, run:

```bash
python main.py --train --num_merges 10 --context_size 20 --nb_layers 6 --training_iterations 5000

### 🤖 Run Inference (Prediction Mode)

Once training is complete, you can load a checkpoint and generate text predictions directly.

```bash
python main.py --ckpt_save_path llm_friends.pt


## ⚙️ Command-Line Arguments

All key parameters can be configured through command-line arguments, making it easy to experiment with different setups.

| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--num_merges` | int | 10 | Number of BPE merges when building the tokenizer |
| `--tokenizer_fraction` | float | 0.1 | Fraction of corpus used to build the tokenizer |
| `--tokenizer_path` | str | `vocabulary.json` | Path to save or load the tokenizer |
| `--context_size` | int | 20 | Context window length for the Transformer |
| `--nb_layers` | int | 6 | Number of Transformer layers |
| `--training_iterations` | int | 5000 | Total training iterations |
| `--eval_interval` | int | 50 | Interval (in steps) between evaluations/checkpoints |
| `--data_path` | str | `Friends_Transcript/Friends_Transcript.txt` | Path to dataset file |
| `--ckpt_save_path` | str | `llm_friends.pt` | Path to save or load model checkpoint |
| `--train` | flag | False | Whether to start/continue training (if omitted → prediction mode) |





