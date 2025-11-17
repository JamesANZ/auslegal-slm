#!/usr/bin/env python3
"""
Train Legal SLM

Fine-tunes DistilGPT2 on the Australian legal corpus using causal language modeling.
"""

import os
import json
import torch
from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset


class LegalDataset(Dataset):
    """Dataset class for legal document sequences."""
    
    def __init__(self, examples: list):
        """
        Initialize dataset.
        
        Args:
            examples: List of tokenized examples with 'input_ids' and 'attention_mask'
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(example['input_ids'], dtype=torch.long)  # For CLM, labels = input_ids
        }


def load_preprocessed_data(data_dir: str):
    """
    Load preprocessed training and validation data.
    
    Args:
        data_dir: Directory containing train_data.json and val_data.json
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    train_path = os.path.join(data_dir, 'train_data.json')
    val_path = os.path.join(data_dir, 'val_data.json')
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_examples = json.load(f)
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_examples = json.load(f)
    
    return train_examples, val_examples


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()


def main():
    """Main training function."""
    # Configuration
    MODEL_NAME = "distilgpt2"
    PREPROCESSED_DATA_DIR = "preprocessed_data"
    OUTPUT_DIR = "models/legal_slm"
    MAX_SEQUENCE_LENGTH = 512
    
    # Training hyperparameters
    # Using 1 epoch for quick end-to-end test (change to 5 for full training)
    NUM_EPOCHS = 1
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
    WARMUP_STEPS = 100
    LOGGING_STEPS = 50
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    
    print("=" * 80)
    print("Legal SLM Training")
    print("=" * 80)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cpu":
        print("  WARNING: Training on CPU will be slow. Consider using GPU if available.")
    
    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Resize token embeddings if needed (shouldn't be needed for GPT-2)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Load preprocessed data
    print(f"\nLoading preprocessed data from {PREPROCESSED_DATA_DIR}...")
    if not os.path.exists(PREPROCESSED_DATA_DIR):
        print(f"ERROR: {PREPROCESSED_DATA_DIR} not found!")
        print("  Please run prepare_data.py first.")
        return
    
    train_examples, val_examples = load_preprocessed_data(PREPROCESSED_DATA_DIR)
    
    print(f"  Training examples: {len(train_examples):,}")
    print(f"  Validation examples: {len(val_examples):,}")
    
    # Create datasets
    train_dataset = LegalDataset(train_examples)
    val_dataset = LegalDataset(val_examples)
    
    # Data collator for language modeling
    # mlm=False means causal language modeling (not masked LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print()
    
    train_result = trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Training summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_result = trainer.evaluate()
    eval_loss = eval_result['eval_loss']
    perplexity = compute_perplexity(eval_loss)
    
    print(f"Validation loss: {eval_loss:.4f}")
    print(f"Validation perplexity: {perplexity:.2f}")
    
    # Save training metrics
    metrics = {
        'training_loss': train_result.training_loss,
        'eval_loss': eval_loss,
        'perplexity': perplexity,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

