# Australian Legal Small Language Model (SLM)

A tiny domain-specific Small Language Model fine-tuned on Australian legal documents scraped from AustLII. This model learns patterns from the legal corpus and generates answers using what it learned during training, **biasing its behavior strongly toward Australian legal language and concepts**.

## Overview

This project implements a Small Language Model (SLM) by fine-tuning DistilGPT2 (~82M parameters) on a corpus of Australian legal documents. The model is trained using causal language modeling, where it learns to predict the next token in sequences of legal text, thereby internalizing domain-specific knowledge, terminology, and reasoning patterns.

### What This Is

- ✅ **A true SLM**: Fine-tuning DistilGPT2 on the legal corpus _is_ training a small language model - the model weights are updated during training
- ✅ **Domain-adapted**: Domain knowledge is internalized in the model parameters (unlike RAG which relies on external document retrieval)
- ✅ **Standalone**: This is fundamentally different from RAG - no external retrieval at inference time
- ✅ **Learns patterns**: The model learns legal terminology, citation patterns, reasoning structures, and domain-specific syntax from the training corpus

### What This Is Not

- ⚠️ **Not trained from scratch**: DistilGPT2 still retains its original pre-training on a large generic corpus; fine-tuning adapts it to the legal domain rather than replacing its prior knowledge
- ⚠️ **Not hallucination-proof**: Fine-tuning _reduces_ general-domain hallucinations by biasing toward legal text, but does **not** prevent hallucinations entirely
- ⚠️ **Not a replacement for legal advice**: This is a research/educational tool for exploring domain-specific language modeling

## Architecture

### Model

- **Base Model**: `distilgpt2` (82M parameters) - the smallest GPT-2 variant
- **Training Objective**: Causal language modeling (next token prediction)
- **Fine-tuning**: Domain adaptation on Australian legal corpus
- **Architecture**: Transformer decoder (GPT-2 style)

### Training Process

1. **Data Preparation**: Legal documents are loaded, cleaned, and tokenized into fixed-length sequences (512 tokens)
2. **Fine-tuning**: DistilGPT2 is fine-tuned using Hugging Face Trainer API with causal language modeling
3. **Evaluation**: Model is evaluated on a held-out validation set to compute perplexity
4. **Inference**: Trained model generates responses to legal questions

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (CPU or GPU)
- CUDA (optional, for GPU training)

### Setup

1. Clone or navigate to this repository:

```bash
cd auslegal-slm
```

2. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 0: Clean Data Files (One-Time)

Clean the raw legal documents by removing metadata headers and irrelevant content:

```bash
python clean_data.py
```

This script:

- Processes all `.txt` files in the `data/` directory
- Strips metadata headers (URL, scraped date, separators)
- Removes navigation/UI elements and irrelevant text
- Cleans and normalizes whitespace
- Saves cleaned versions back to the same files

**Note**: This is a one-time operation. Once files are cleaned, you don't need to run this again unless you scrape new data.

### Step 1: Prepare Data

Preprocess the cleaned legal documents for training:

```bash
python prepare_data.py
```

This script:

- Loads all cleaned `.txt` files from the `data/` directory
- Tokenizes documents into fixed-length sequences (512 tokens)
- Splits into training (90%) and validation (10%) sets
- Saves preprocessed data to `preprocessed_data/`

**Output**: `preprocessed_data/train_data.json` and `preprocessed_data/val_data.json`

### Step 2: Train the Model

Fine-tune DistilGPT2 on the legal corpus:

```bash
python train_slm.py
```

**Training Configuration**:

- Epochs: 5
- Learning rate: 2e-5
- Batch size: 4 (effective: 16 with gradient accumulation)
- Max sequence length: 512 tokens
- Optimizer: AdamW with warmup

**Output**: Trained model saved to `models/legal_slm/`

**Training Time**:

- CPU: Several hours (depending on corpus size)
- GPU: 30-60 minutes (depending on GPU)

### Step 3: Query the Model

#### Interactive Mode

Run the interactive query interface:

```bash
python query_slm.py
```

Then enter legal questions interactively. Type `quit` or `exit` to stop.

#### Single Question

Ask a single question:

```bash
python query_slm.py --question "What is the legal precedent for negligence in Australian law?"
```

#### Custom Parameters

Adjust generation parameters:

```bash
python query_slm.py \
    --question "Your question here" \
    --temperature 0.3 \
    --max-length 300
```

**Parameters**:

- `--model-dir`: Path to fine-tuned model (default: `models/legal_slm`)
- `--question`: Single question to ask
- `--temperature`: Sampling temperature, 0.0-1.0 (lower = more deterministic, default: 0.4)
- `--max-length`: Maximum response length in tokens (default: 250)

## Technical Specifications (NatSpec)

### Data Format

Legal documents are stored as plain text files with the following structure:

```
URL: https://www.austlii.edu.au/...
Scraped: YYYY-MM-DD HH:MM:SS
================================================================================

[Legal content here]
```

The data preparation script automatically strips the metadata header and extracts only the legal content.

### Tokenization

- **Tokenizer**: GPT-2 tokenizer (BPE-based)
- **Vocabulary size**: 50,257 tokens
- **Special tokens**: `<|endoftext|>` (EOS), padding token set to EOS
- **Sequence length**: 512 tokens (fixed)
- **Sliding window**: 256 token stride (50% overlap)

### Model Architecture

- **Architecture**: GPT-2 (Transformer decoder)
- **Parameters**: ~82M (DistilGPT2)
- **Layers**: 6 transformer decoder blocks
- **Hidden size**: 768
- **Attention heads**: 12
- **Max position embeddings**: 1024

### Training Configuration

```python
Model: distilgpt2
Objective: Causal Language Modeling (CLM)
Loss: Cross-entropy
Optimizer: AdamW
Learning rate: 2e-5
Learning rate schedule: Linear warmup + cosine decay
Batch size: 4 (per device)
Gradient accumulation: 4 steps (effective batch: 16)
Epochs: 5
Warmup steps: 100
Max sequence length: 512 tokens
Mixed precision: FP16 (if GPU available)
```

### Inference Configuration

```python
Temperature: 0.4 (default, range: 0.0-1.0)
Top-p (nucleus): 0.9
Top-k: 50
Repetition penalty: 1.2
Max new tokens: 250
Do sample: True
```

## Limitations and Considerations

### Hallucination Mitigation

The following strategies are employed to reduce hallucinations and off-domain content:

- **Domain fine-tuning**: Model is fine-tuned only on the legal corpus (though base model retains general pre-training)
- **Low temperature**: 0.3-0.5 during inference to reduce randomness
- **Capped generation length**: Limits response length to prevent rambling
- **Prompt engineering**: Prompts explicitly reference "Australian legal documents"
- **Manual monitoring**: Test prompts should be used to detect off-domain or invented content

**Important Note**: Fine-tuning and decoding settings _reduce_ hallucinations and off-domain content but **cannot guarantee their absence**. The model may still:

- Generate plausible-sounding but incorrect legal information
- Mix general knowledge with legal domain knowledge
- Produce responses that don't directly cite the training corpus

For stricter factual grounding, consider pairing this SLM with a RAG (Retrieval-Augmented Generation) pipeline over the same corpus.

### Data Limitations

- **Corpus size**: Training on 119 documents (or ~856 files as scraped) is relatively small for language model training
- **Coverage**: The model may not have seen all areas of Australian law
- **Temporal**: Documents reflect the state of law at scraping time; laws may have changed

### Model Limitations

- **Context window**: 512 tokens limits the amount of context the model can consider
- **Generalization**: May overfit to specific documents or underperform on unseen legal topics
- **No citations**: Model doesn't explicitly cite sources (unlike RAG systems)

## Evaluation

Training metrics are saved to `models/legal_slm/training_metrics.json`:

```json
{
  "training_loss": 2.3456,
  "eval_loss": 2.4567,
  "perplexity": 11.67,
  "num_epochs": 5,
  "learning_rate": 2e-5,
  "batch_size": 4
}
```

**Perplexity**: Lower is better. Measures how well the model predicts the next token. A perplexity of ~10-15 is reasonable for domain-adapted models.

## Future Enhancements

### Comparison Models

For learning and comparison purposes, additional training approaches can be implemented:

- **N-gram model** (`train_ngram.py`): Classic n-gram language model trained from scratch
- **Char-RNN** (`train_charrnn.py`): Character-level LSTM/GRU trained from scratch
- **Tiny transformer from scratch**: Fully custom transformer trained only on legal corpus

These can be compared on:

- Training time
- Validation loss/perplexity
- Qualitative sample quality
- Memory requirements

### Hybrid Approaches

- **SLM + RAG**: Combine fine-tuned SLM with retrieval over the same corpus for stricter factual grounding
- **LoRA fine-tuning**: More parameter-efficient fine-tuning approach
- **Gradient checkpointing**: Reduce memory usage for larger batch sizes

## File Structure

```
auslegal-slm/
├── data/                    # Legal documents (scraped, cleaned)
├── preprocessed_data/       # Tokenized training data
│   ├── train_data.json
│   └── val_data.json
├── models/                  # Trained models
│   └── legal_slm/          # Fine-tuned DistilGPT2
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.json
│       ├── merges.txt
│       └── training_metrics.json
├── scraper/                 # Data collection tools
│   ├── scraper.py          # Legal document scraper
│   └── requirements.txt    # Scraper dependencies
├── clean_data.py           # One-time data cleaning script
├── prepare_data.py         # Data preparation script
├── train_slm.py           # Training script
├── query_slm.py           # Query interface
├── requirements.txt       # SLM dependencies
└── README.md              # This file
```

## Citation

If you use this code or model, please cite:

```bibtex
@software{auslegal_slm,
  title = {Australian Legal Small Language Model},
  author = {James Sangalli},
  year = {2025},
  url = {https://github.com/JamesANZ/auslegal-slm}
}
```

## Donate

If you find this project useful, consider supporting it with Bitcoin:

**⚡ Lightning Network**

<img src="https://raw.githubusercontent.com/bitcoinwarrior1/CitySats/main/public/lightning.jpeg" alt="Lightning QR Code" width="120" />

<code>lnbc1pjhhsqepp5mjgwnvg0z53shm22hfe9us289lnaqkwv8rn2s0rtekg5vvj56xnqdqqcqzzsxqyz5vqsp5gu6vh9hyp94c7t3tkpqrp2r059t4vrw7ps78a4n0a2u52678c7yq9qyyssq7zcferywka50wcy75skjfrdrk930cuyx24rg55cwfuzxs49rc9c53mpz6zug5y2544pt8y9jflnq0ltlha26ed846jh0y7n4gm8jd3qqaautqa</code>

**₿ On-Chain**

<img src="https://raw.githubusercontent.com/bitcoinwarrior1/CitySats/main/public/onchain.jpg" alt="Bitcoin Address QR Code" width="120" />

<code>bc1ptzvr93pn959xq4et6sqzpfnkk2args22ewv5u2th4ps7hshfaqrshe0xtp</code>

## Acknowledgments

- Legal documents scraped from [AustLII](https://www.austlii.edu.au/)
- Model architecture based on [DistilGPT2](https://huggingface.co/distilgpt2) by Hugging Face
- Built with [Transformers](https://huggingface.co/docs/transformers) library
