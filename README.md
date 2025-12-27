# Australian Legal Small Language Model (SLM)

A domain-specific Small Language Model fine-tuned on Australian legal documents. The model learns patterns from AustLII legal corpus and generates answers using domain-adapted knowledge, with strong bias toward Australian legal language and concepts.

A Small Language Model (SLM) fine-tuned from DistilGPT2 (~82M parameters) on Australian legal documents scraped from AustLII. The model learns legal terminology, citation patterns, and reasoning structures through causal language modeling.

## Features

- **Domain-Adapted**: Fine-tuned specifically on Australian legal documents
- **Learns Patterns**: Internalizes legal terminology, citations, and reasoning structures
- **Standalone**: No external retrieval needed (unlike RAG systems)
- **Lightweight**: 82M parameters, runs efficiently on CPU or GPU

## What This Is

- A true SLM: Fine-tuning DistilGPT2 on legal corpus updates model weights
- Domain-adapted: Knowledge internalized in model parameters
- Standalone: No external retrieval at inference time
- Learns patterns: Legal terminology, citations, reasoning structures

## What This Is Not

- Not trained from scratch: Built on DistilGPT2's pre-training
- Not hallucination-proof: Reduces but doesn't eliminate hallucinations
- Not legal advice: Research/educational tool only

## Quick Start

### Installation

**Requirements:** Python 3.8+, PyTorch, CUDA (optional for GPU)

```bash
# Clone repository
git clone https://github.com/JamesANZ/auslegal-slm.git
cd auslegal-slm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training Pipeline

1. **Clean Data** (one-time):
```bash
python clean_data.py
```

2. **Prepare Data**:
```bash
python prepare_data.py
```

3. **Train Model**:
```bash
python train_slm.py
```

**Training Time:**
- CPU: Several hours
- GPU: 30-60 minutes

### Query the Model

**Interactive Mode:**
```bash
python query_slm.py
```

**Single Question:**
```bash
python query_slm.py --question "What is negligence?"
```

**Best Prompting Strategy:**

Since the model was trained on raw legal documents (not Q&A pairs), it works best when prompts are framed as legal text continuations. The default continuation strategy automatically converts questions to legal definition prompts.

**Recommended Settings:**
```bash
python query_slm.py \
    --question "What is negligence?" \
    --strategy continuation \
    --greedy \
    --max-length 300
```

**Available Strategies:**
- `continuation` (default, recommended): Converts questions to legal definition prompts
- `few_shot`: Shows examples of desired format
- `direct`: Simple Q&A format
- `structured`: Uses XML-like delimiters

**Python API:**
```python
from query_slm import LegalSLM, PromptStrategy

slm = LegalSLM()
answer = slm.generate_answer(
    "What is negligence?",
    use_greedy=True,              # Most deterministic
    max_length=300,               # Complete answers
    strategy=PromptStrategy.CONTINUATION,
    stop_sequences=["\n\n", ".", "?"]
)
```

## Architecture

### Model
- **Base Model**: `distilgpt2` (82M parameters)
- **Architecture**: Transformer decoder (GPT-2 style)
- **Training Objective**: Causal language modeling (next token prediction)
- **Layers**: 6 transformer decoder blocks
- **Hidden size**: 768
- **Attention heads**: 12

### Training Configuration

```python
Epochs: 5
Learning rate: 2e-5
Batch size: 4 (effective: 16 with gradient accumulation)
Max sequence length: 512 tokens
Optimizer: AdamW with warmup
Mixed precision: FP16 (if GPU available)
```

### Inference Configuration

**Recommended Settings:**
```python
Strategy: continuation (converts questions to legal definition prompts)
Decoding: greedy (most deterministic)
Max length: 300 tokens (complete answers)
Temperature: 0.2 (when using sampling)
Repetition penalty: 1.2
```

**Why Continuation Strategy Works Best:**
The model was trained on raw legal documents, not Q&A pairs. Framing questions as legal text continuations (e.g., "In Australian law, negligence is defined as") matches the training data format and produces better results than Q&A style prompts.

## Data

### Source
- **Legal documents** scraped from [AustLII](https://www.austlii.edu.au/)
- **Format**: Plain text files with metadata headers
- **Processing**: Automatic cleaning and tokenization

### Tokenization
- **Tokenizer**: GPT-2 tokenizer (BPE-based)
- **Vocabulary size**: 50,257 tokens
- **Sequence length**: 512 tokens (fixed)
- **Sliding window**: 256 token stride (50% overlap)

## Limitations

### Hallucination Mitigation
- Domain fine-tuning on legal corpus only
- Greedy decoding or low temperature (0.2-0.3) during inference
- Capped generation length (300 tokens default)
- Continuation strategy prompts that match training data format

**Note**: Fine-tuning reduces but **cannot guarantee** absence of hallucinations. The model may still generate incorrect or mixed-domain content.

### Data Limitations
- **Corpus size**: Training on 119 documents is relatively small
- **Coverage**: May not cover all areas of Australian law
- **Temporal**: Documents reflect law at scraping time

### Model Limitations
- **Context window**: 512 tokens limits context
- **Generalization**: May overfit to specific documents
- **No citations**: Doesn't explicitly cite sources (unlike RAG)

## Evaluation

Training metrics saved to `models/legal_slm/training_metrics.json`:

```json
{
  "training_loss": 2.3456,
  "eval_loss": 2.4567,
  "perplexity": 11.67,
  "num_epochs": 5
}
```

**Perplexity**: Lower is better. ~10-15 is reasonable for domain-adapted models.

## Use Cases

- **Research** – Explore domain-specific language modeling
- **Education** – Learn about fine-tuning and SLM training
- **Prototyping** – Test legal domain adaptation approaches
- **Comparison** – Baseline for hybrid SLM+RAG systems

## Future Enhancements

- **SLM + RAG**: Combine with retrieval for stricter factual grounding
- **LoRA fine-tuning**: More parameter-efficient approach
- **Comparison models**: N-gram, Char-RNN, tiny transformer from scratch
- **Gradient checkpointing**: Reduce memory for larger batches

## Project Structure

```
auslegal-slm/
├── data/                    # Legal documents (scraped, cleaned)
├── preprocessed_data/       # Tokenized training data
├── models/                  # Trained models
│   └── legal_slm/          # Fine-tuned DistilGPT2
├── scraper/                 # Data collection tools
├── clean_data.py           # Data cleaning script
├── prepare_data.py         # Data preparation script
├── train_slm.py           # Training script
├── query_slm.py           # Query interface
└── requirements.txt       # Dependencies
```

## Citation

```bibtex
@software{auslegal_slm,
  title = {Australian Legal Small Language Model},
  author = {James Sangalli},
  year = {2025},
  url = {https://github.com/JamesANZ/auslegal-slm}
}
```

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Legal documents from [AustLII](https://www.austlii.edu.au/)
- Model architecture: [DistilGPT2](https://huggingface.co/distilgpt2) by Hugging Face
- Built with [Transformers](https://huggingface.co/docs/transformers) library

## Support

If you find this project useful, consider supporting it:

**Lightning Network:**
```
lnbc1pjhhsqepp5mjgwnvg0z53shm22hfe9us289lnaqkwv8rn2s0rtekg5vvj56xnqdqqcqzzsxqyz5vqsp5gu6vh9hyp94c7t3tkpqrp2r059t4vrw7ps78a4n0a2u52678c7yq9qyyssq7zcferywka50wcy75skjfrdrk930cuyx24rg55cwfuzxs49rc9c53mpz6zug5y2544pt8y9jflnq0ltlha26ed846jh0y7n4gm8jd3qqaautqa
```

**Bitcoin**: [bc1ptzvr93pn959xq4et6sqzpfnkk2args22ewv5u2th4ps7hshfaqrshe0xtp](https://mempool.space/address/bc1ptzvr93pn959xq4et6sqzpfnkk2args22ewv5u2th4ps7hshfaqrshe0xtp)

**Ethereum/EVM**: [0x42ea529282DDE0AA87B42d9E83316eb23FE62c3f](https://etherscan.io/address/0x42ea529282DDE0AA87B42d9E83316eb23FE62c3f)
