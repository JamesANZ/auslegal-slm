# Legal RAG System

A Retrieval-Augmented Generation (RAG) system for querying Australian legal documents using LangChain and multiple LLM providers.

## Overview

This RAG system processes ~10,637 legal text files from the `data/` directory and provides query-based information retrieval. It uses:

- **LangChain** for document processing, embeddings, and retrieval
- **FAISS** for efficient vector storage and similarity search
- **sentence-transformers** for local embeddings (no API key required)
- **Multiple LLM providers** from [cross-llm-mcp](https://github.com/JamesANZ/cross-llm-mcp) for answer generation

### Supported LLM Providers

The system supports 8 LLM providers, matching the providers in cross-llm-mcp:

1. **ChatGPT** (OpenAI) - `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
2. **Claude** (Anthropic) - `claude-3-sonnet-20240229`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`
3. **DeepSeek** - `deepseek-chat`, `deepseek-coder`
4. **Gemini** (Google) - `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`
5. **Grok** (xAI) - `grok-3`
6. **Kimi** (Moonshot AI) - `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`
7. **Perplexity** - `sonar-pro`, `sonar-medium-online`, `sonar-small-online`
8. **Mistral** - `mistral-large-latest`, `mistral-small-latest`, `mixtral-8x7b-32768`

## Installation

### 1. Install Dependencies

```bash
pip install -r rag/requirements.txt
```

Or install in the existing virtual environment:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r rag/requirements.txt
```

### 2. Set Up API Keys

You need to provide an API key for at least one LLM provider. You can set API keys via:

**Option A: Environment Variables**

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
GEMINI_API_KEY=your-gemini-key
XAI_API_KEY=your-grok-key
KIMI_API_KEY=your-kimi-key
PERPLEXITY_API_KEY=your-perplexity-key
MISTRAL_API_KEY=your-mistral-key
```

**Option B: CLI Parameter**

Pass the API key directly via `--api-key` parameter (see Usage section).

### Getting API Keys

#### OpenAI (ChatGPT)
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Create a new API key
4. Set `OPENAI_API_KEY` environment variable

#### Anthropic (Claude)
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Create a new API key
4. Set `ANTHROPIC_API_KEY` environment variable

#### DeepSeek
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Sign up or log in
3. Create a new API key
4. Set `DEEPSEEK_API_KEY` environment variable

#### Google Gemini
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign up or log in with Google account
3. Create a new API key
4. Set `GEMINI_API_KEY` environment variable

#### xAI (Grok)
1. Visit [xAI Platform](https://console.x.ai/)
2. Sign up or log in
3. Create a new API key
4. Set `XAI_API_KEY` environment variable

#### Moonshot AI (Kimi)
1. Visit [Moonshot AI Platform](https://platform.moonshot.ai/)
2. Sign up or log in
3. Create a new API key
4. Set `KIMI_API_KEY` or `MOONSHOT_API_KEY` environment variable

#### Perplexity AI
1. Visit [Perplexity AI Platform](https://www.perplexity.ai/hub)
2. Sign up or log in
3. Generate a new API key from developer console
4. Set `PERPLEXITY_API_KEY` environment variable

#### Mistral AI
1. Visit [Mistral AI Console](https://console.mistral.ai/)
2. Sign up or log in
3. Create a new API key
4. Set `MISTRAL_API_KEY` environment variable

## Usage

### Step 1: Build the Vector Store Index

First, build the vector store index from your legal documents:

```bash
python rag/build_index.py
```

This will:
- Load all `.txt` files from the `data/` directory
- Apply cleaning to remove metadata and irrelevant content
- Split documents into chunks (1000 chars, 200 overlap)
- Generate embeddings using sentence-transformers (local, no API key needed)
- Save the FAISS vector store to `rag/vectorstore/`

**Options:**
```bash
python rag/build_index.py --help

# Custom data directory
python rag/build_index.py --data-dir /path/to/data

# Custom chunk size
python rag/build_index.py --chunk-size 1500 --chunk-overlap 300

# Force rebuild
python rag/build_index.py --force-rebuild
```

### Step 2: Query the RAG System

#### Interactive Mode

Run the interactive query interface:

```bash
python rag/query_rag.py --llm chatgpt --api-key sk-your-key
```

Then enter legal questions interactively. Type `quit` or `exit` to stop.

**Interactive Commands:**
- `quit`, `exit`, `q` - Exit the interface
- `llm <provider>` - Switch LLM provider (e.g., `llm claude`)
- `model <model>` - Change model (e.g., `model gpt-4-turbo`)

#### Single Question

Ask a single question:

```bash
python rag/query_rag.py \
    --llm claude \
    --api-key sk-ant-your-key \
    --question "What is the legal precedent for negligence in Australian law?"
```

#### With Source Documents

Show source documents with answers:

```bash
python rag/query_rag.py \
    --llm gemini \
    --api-key your-gemini-key \
    --question "Explain the concept of duty of care in Australian tort law" \
    --show-sources
```

Show full content of source documents:

```bash
python rag/query_rag.py \
    --llm chatgpt \
    --api-key sk-your-key \
    --question "Your question" \
    --show-sources \
    --show-full-content
```

#### Custom Parameters

Adjust retrieval and generation parameters:

```bash
python rag/query_rag.py \
    --llm mistral \
    --api-key your-mistral-key \
    --question "Your question" \
    --temperature 0.5 \
    --max-tokens 2000 \
    --top-k 6
```

### Command-Line Options

#### `query_rag.py` Options

- `--llm` - LLM provider (chatgpt, claude, deepseek, gemini, grok, kimi, perplexity, mistral)
- `--model` - Specific model name (uses provider default if not specified)
- `--api-key` - API key for the LLM provider (overrides environment variable)
- `--question` - Single question to ask (runs interactive mode if not provided)
- `--vectorstore-path` - Path to vector store directory (default: `rag/vectorstore`)
- `--temperature` - Sampling temperature 0.0-2.0 (default: 0.7)
- `--max-tokens` - Maximum tokens in response (default: 1000)
- `--top-k` - Number of documents to retrieve (default: 4)
- `--show-sources` - Show source documents with answers
- `--show-full-content` - Show full content of source documents

#### `build_index.py` Options

- `--data-dir` - Path to data directory with legal documents (default: `data`)
- `--vectorstore-path` - Path to vector store directory (default: `rag/vectorstore`)
- `--chunk-size` - Size of text chunks (default: 1000)
- `--chunk-overlap` - Overlap between chunks (default: 200)
- `--force-rebuild` - Force rebuild of index even if it exists

## Examples

### Example 1: Query with ChatGPT

```bash
python rag/query_rag.py \
    --llm chatgpt \
    --api-key $OPENAI_API_KEY \
    --question "What are the key elements of a contract in Australian law?" \
    --show-sources
```

### Example 2: Query with Claude

```bash
python rag/query_rag.py \
    --llm claude \
    --model claude-3-opus-20240229 \
    --api-key $ANTHROPIC_API_KEY \
    --question "Explain the doctrine of precedent in Australian courts" \
    --temperature 0.3 \
    --top-k 6
```

### Example 3: Interactive Mode with Multiple LLMs

```bash
# Start with ChatGPT
python rag/query_rag.py --llm chatgpt --api-key $OPENAI_API_KEY

# Then switch LLMs in interactive mode:
# > llm claude
# > llm gemini
# > llm mistral
```

### Example 4: Using Environment Variables

```bash
# Set API key in environment
export ANTHROPIC_API_KEY=sk-ant-your-key

# Query without --api-key
python rag/query_rag.py \
    --llm claude \
    --question "Your question"
```

## Configuration

### Default Settings

Configuration is defined in `rag/config.py`:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (local, no API key)
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Top K Retrieval**: 4 documents
- **Default Temperature**: 0.7
- **Default Max Tokens**: 1000
- **Vector Store Path**: `rag/vectorstore/`
- **Data Directory**: `data/`

### Default Models per Provider

Matching cross-llm-mcp defaults:

- ChatGPT: `gpt-4`
- Claude: `claude-3-sonnet-20240229`
- DeepSeek: `deepseek-chat`
- Gemini: `gemini-2.5-flash`
- Grok: `grok-3`
- Kimi: `moonshot-v1-8k`
- Perplexity: `sonar-pro`
- Mistral: `mistral-large-latest`

## Architecture

### Document Processing Pipeline

1. **Load**: `DirectoryLoader` loads all `.txt` files from `data/`
2. **Clean**: Apply `clean_legal_text()` to remove metadata and irrelevant content
3. **Split**: `RecursiveCharacterTextSplitter` splits into chunks (1000 chars, 200 overlap)
4. **Embed**: `HuggingFaceEmbeddings` generates embeddings locally
5. **Store**: FAISS vector store persists to `rag/vectorstore/`

### Query Pipeline

1. **Retrieve**: FAISS similarity search retrieves top-k relevant document chunks
2. **Context**: Retrieved chunks are combined into context
3. **Generate**: Selected LLM generates answer based on context
4. **Return**: Answer with source documents and metadata

### LLM Integration

The system uses LangChain's native integrations:

- **ChatGPT**: `langchain-openai.ChatOpenAI`
- **Claude**: `langchain-anthropic.ChatAnthropic`
- **Gemini**: `langchain-google-genai.ChatGoogleGenerativeAI`
- **Mistral**: `langchain-mistralai.ChatMistralAI`
- **DeepSeek, Grok, Kimi, Perplexity**: `langchain-openai.ChatOpenAI` with custom `base_url`

## File Structure

```
rag/
├── __init__.py              # Package initialization
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── config.py               # Configuration and LLM provider settings
├── rag_system.py           # Core LegalRAG class
├── build_index.py          # Script to build vector store
├── query_rag.py            # CLI interface for querying
└── vectorstore/            # FAISS vector store (generated)
    ├── index.faiss
    └── index.pkl
```

## Troubleshooting

### Vector Store Not Found

```
ERROR: Vector store not found at rag/vectorstore
```

**Solution**: Run `python rag/build_index.py` first to build the index.

### API Key Not Found

```
ERROR: ChatGPT API key not found
```

**Solution**: Set the appropriate environment variable or use `--api-key` parameter.

### Import Errors

```
ModuleNotFoundError: No module named 'langchain_community'
```

**Solution**: Install dependencies with `pip install -r rag/requirements.txt`

### No Documents Found

```
ERROR: No documents found in data/
```

**Solution**: Ensure the `data/` directory contains `.txt` files with legal documents.

### LLM Provider Not Supported

```
ERROR: Unsupported LLM provider: xyz
```

**Solution**: Use one of the supported providers: `chatgpt`, `claude`, `deepseek`, `gemini`, `grok`, `kimi`, `perplexity`, `mistral`

## Limitations

- **API Costs**: Each query uses the selected LLM's API, which may incur costs
- **Embedding Quality**: Uses local sentence-transformers model; may not be as good as OpenAI embeddings
- **Context Window**: Limited by the LLM's context window and retrieved document chunks
- **Hallucination**: LLMs may still hallucinate; always verify answers against source documents
- **Not Legal Advice**: This is a research tool and should not be used for legal advice

## Integration with Existing SLM

This RAG system is separate from the existing SLM (Small Language Model) in `models/legal_slm/`:

- **SLM**: Fine-tuned DistilGPT2 that generates text based on learned patterns
- **RAG**: Retrieves relevant documents and uses external LLMs to answer questions

The RAG system provides:
- ✅ Source attribution (cites specific documents)
- ✅ Better factual grounding (retrieves actual documents)
- ✅ Multiple LLM options (8 providers)
- ✅ No training required (uses pre-built index)

## Citation

If you use this RAG system, please cite:

```bibtex
@software{auslegal_rag,
  title = {Australian Legal RAG System},
  author = {James Sangalli},
  year = {2025},
  url = {https://github.com/JamesANZ/auslegal-slm}
}
```

## Acknowledgments

- Legal documents from [AustLII](https://www.austlii.edu.au/)
- Built with [LangChain](https://www.langchain.com/)
- LLM providers from [cross-llm-mcp](https://github.com/JamesANZ/cross-llm-mcp)
- Embeddings from [sentence-transformers](https://www.sbert.net/)
- Vector store using [FAISS](https://github.com/facebookresearch/faiss)

