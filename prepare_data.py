#!/usr/bin/env python3
"""
Data Preparation for Legal SLM Training

Loads and preprocesses legal documents from the data directory for training.
Strips metadata headers and prepares tokenized sequences for causal language modeling.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple
from transformers import GPT2Tokenizer


def load_legal_documents(data_dir: str) -> List[str]:
    """
    Load all legal document text files from the data directory.
    
    Files should already be cleaned by clean_data.py (metadata headers removed).
    This function just loads the cleaned content.
    
    Args:
        data_dir: Path to directory containing .txt files
        
    Returns:
        List of document texts
    """
    documents = []
    data_path = Path(data_dir)
    
    # Get all .txt files (excluding hidden files like .scraper_progress.json)
    txt_files = sorted([f for f in data_path.glob("*.txt") if not f.name.startswith('.')])
    
    print(f"Found {len(txt_files)} text files in {data_dir}")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().strip()
            
            # Files should already be cleaned, so just load the content
            # Skip very short documents (likely errors or empty)
            if len(content) > 100:
                documents.append(content)
            else:
                print(f"  Skipping short document: {file_path.name} ({len(content)} chars)")
                
        except UnicodeDecodeError as e:
            print(f"  Error decoding {file_path.name}: {e} (skipping)")
            continue
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e} (skipping)")
            continue
    
    print(f"Loaded {len(documents)} valid documents")
    return documents


def clean_text(text: str) -> str:
    """
    Light cleaning of text content (files should already be cleaned by clean_data.py).
    
    This is just a safety pass for any remaining issues.
    
    Args:
        text: Text content (should already be cleaned)
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace (safety check)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    
    return text.strip()


def tokenize_documents(
    documents: List[str],
    tokenizer: GPT2Tokenizer,
    max_length: int = 512,
    stride: int = 256
) -> List[dict]:
    """
    Tokenize documents into fixed-length sequences for training.
    
    Uses sliding window approach to create overlapping sequences.
    
    Args:
        documents: List of document texts
        tokenizer: GPT-2 tokenizer instance
        max_length: Maximum sequence length in tokens
        stride: Number of tokens to slide window (overlap)
        
    Returns:
        List of tokenized examples with 'input_ids' and 'attention_mask'
    """
    tokenized_examples = []
    
    print(f"Tokenizing documents with max_length={max_length}, stride={stride}...")
    
    for doc_idx, doc in enumerate(documents):
        # Clean the document
        cleaned_doc = clean_text(doc)
        
        # Tokenize the document (truncate if very long to avoid warnings)
        # We'll create windows anyway, so truncating here is safe
        max_doc_tokens = 100000  # Reasonable limit before windowing
        tokens = tokenizer.encode(
            cleaned_doc, 
            add_special_tokens=True,
            max_length=max_doc_tokens,
            truncation=True
        )
        
        # Create sliding windows
        for i in range(0, len(tokens), stride):
            window = tokens[i:i + max_length]
            
            # Pad if necessary (shouldn't happen with stride, but safety check)
            if len(window) < max_length:
                # Only add if it's the last window and has reasonable length
                if len(window) > 50:  # Minimum meaningful length
                    padding_length = max_length - len(window)
                    # Ensure pad_token_id is set
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    window = window + [pad_token_id] * padding_length
                else:
                    break
            
            tokenized_examples.append({
                'input_ids': window,
                'attention_mask': [1] * max_length
            })
        
        if (doc_idx + 1) % 10 == 0:
            print(f"  Processed {doc_idx + 1}/{len(documents)} documents ({len(tokenized_examples)} sequences)")
    
    print(f"Created {len(tokenized_examples)} tokenized sequences")
    return tokenized_examples


def split_train_val(
    examples: List[dict],
    train_ratio: float = 0.9
) -> Tuple[List[dict], List[dict]]:
    """
    Split examples into training and validation sets.
    
    Args:
        examples: List of tokenized examples
        train_ratio: Proportion of data for training (default 0.9 = 90%)
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"Split: {len(train_examples)} training, {len(val_examples)} validation examples")
    return train_examples, val_examples


def save_preprocessed_data(
    train_examples: List[dict],
    val_examples: List[dict],
    output_dir: str
):
    """
    Save preprocessed data to JSON files.
    
    Args:
        train_examples: Training examples
        val_examples: Validation examples
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_data.json')
    val_path = os.path.join(output_dir, 'val_data.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples, f, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_examples, f, indent=2)
    
    print(f"Saved preprocessed data to {output_dir}/")


def main():
    """Main data preparation pipeline."""
    # Configuration
    DATA_DIR = "data"
    OUTPUT_DIR = "preprocessed_data"
    MAX_LENGTH = 512
    STRIDE = 256
    TRAIN_RATIO = 0.9
    
    print("=" * 80)
    print("Legal SLM Data Preparation")
    print("=" * 80)
    
    # Initialize tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Set pad token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load documents
    print(f"\nLoading documents from {DATA_DIR}...")
    documents = load_legal_documents(DATA_DIR)
    
    if not documents:
        print("ERROR: No documents found!")
        return
    
    # Calculate corpus statistics
    total_chars = sum(len(doc) for doc in documents)
    total_words = sum(len(doc.split()) for doc in documents)
    print(f"\nCorpus statistics:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total words: {total_words:,}")
    print(f"  Avg document length: {total_chars // len(documents):,} chars")
    
    # Tokenize documents
    print(f"\nTokenizing documents...")
    tokenized_examples = tokenize_documents(
        documents,
        tokenizer,
        max_length=MAX_LENGTH,
        stride=STRIDE
    )
    
    if not tokenized_examples:
        print("ERROR: No tokenized examples created!")
        return
    
    # Split into train/val
    print(f"\nSplitting into train/validation sets...")
    train_examples, val_examples = split_train_val(tokenized_examples, TRAIN_RATIO)
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data...")
    save_preprocessed_data(train_examples, val_examples, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

