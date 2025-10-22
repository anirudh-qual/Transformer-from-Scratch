# Transformer from Scratch

An educational PyTorch implementation of the original Transformer architecture used for
sequence-to-sequence neural machine translation. The goal of this repository is to show
all of the moving pieces involved in training a transformer model end-to-end on a
bilingual corpus, starting from tokenization and batching all the way to optimization
and experiment tracking.

## Project Structure

```
├── config.py               # Central place for hyperparameters and paths
├── src/
│   ├── dataset.py          # `BilingualDataset` plus dataloader utilities
│   ├── train.py            # Tokenizer creation and the training orchestration logic
│   └── tranformer_model.py # Building blocks that compose the Transformer architecture
└── README.md
```

Key architectural components implemented in `src/tranformer_model.py` include input
embeddings, sinusoidal positional encodings, multi-head attention, encoder/decoder
stacks, and the final projection layer that produces token logits.

## Requirements

The code targets **Python 3.10+** and relies on the following third-party libraries:

- [PyTorch](https://pytorch.org/) for tensor operations, neural network modules, and
  optimization.
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) to download the
  bilingual *opus_books* corpus used for training.
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers) for building
  WordLevel tokenizers with special tokens.
- [TensorBoard](https://www.tensorflow.org/tensorboard) to visualize loss curves and
  other metrics during training.

You can install the dependencies into a virtual environment with:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install torch datasets tokenizers tensorboard
```

> **Note:** Downloading the dataset requires an active internet connection the first
> time you run the training script.

## Configuration

All tunable hyperparameters and file-system paths live in [`config.py`](./config.py).
Some notable values:

- `lang_src` / `lang_tgt`: Language pair pulled from the Hugging Face dataset.
- `seq_len`: Fixed maximum sequence length (including special tokens) used for
  padding/truncation inside the `BilingualDataset` class.
- `batch_size`, `num_epochs`, `lr`: Standard training controls for the optimizer.
- `model_folder`, `model_filename`: Location and naming scheme used when checkpointing
  weights.
- `tokenizer_file`: Template string for saving/loading tokenizer JSON files per
  language.
- `experiment_name`: Folder written by TensorBoard's `SummaryWriter` for logging.

Adjust these values to fit your machine (e.g., shorten `seq_len` or reduce
`batch_size` if GPU memory is limited).

## Running Training

The repository does not ship with a top-level CLI entry point, but you can launch the
training loop by importing `train_model` and feeding it the configuration dictionary.
From the project root:

```bash
PYTHONPATH="src:." python -c "from config import get_config; from train import train_model; train_model(get_config())"
```

This command will:

1. Download (or reuse) the *opus_books* dataset for the configured language pair.
2. Train WordLevel tokenizers for both source and target languages if their cached
   JSON files are not present.
3. Construct PyTorch `DataLoader` instances backed by the `BilingualDataset`, which
   handles adding `[SOS]`, `[EOS]`, and `[PAD]` tokens and generates the masks required
   for attention.
4. Build the Transformer model via `build_transformer` and start the epoch loop with
   Adam optimization and label smoothing applied in the cross-entropy loss.
5. Persist checkpoints and TensorBoard logs using the paths defined in `config.py`.

During training you can monitor progress with TensorBoard:

```bash
tensorboard --logdir runs
```

Open the provided URL in a browser to inspect scalar charts and compare experiments.

## Dataset Details

- **Corpus**: [Helsinki-NLP/opus_books](https://huggingface.co/datasets/opus_books)
- **Split**: `train`
- **Languages**: English to Italian by default (`en-it`), configurable through
  `config.py`
- **Pre-processing**: Word-level tokenization with `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`
  special tokens and dynamic padding/truncation to the configured `seq_len`.

If you wish to experiment with other language pairs exposed by *opus_books*, simply
change the `lang_src`/`lang_tgt` values and delete the cached tokenizer JSON files so
that new tokenizers will be trained.

## Known Gaps & Next Steps

This repository is intentionally minimal and a few pieces are left for the reader to
extend:

- The inner epoch loop in `src/train.py` currently prepares the model and optimizer but
  still needs the step logic that feeds batches, computes loss, performs backpropagation,
  and writes TensorBoard scalars.
- Model export/inference helpers (e.g., greedy decoding, beam search, translation demo)
  are not yet implemented. `tranformer_model.py` exposes the building blocks required
  for those extensions.
- Error handling and checkpoint resumption can be enhanced depending on your needs.

Feel free to fork the project, iterate on these items, and tailor the codebase to your
own experiments.

## References

- Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural Information
  Processing Systems* (2017).
- The excellent "Transformer from Scratch" series of blog posts and code walkthroughs
  that inspired this implementation.
