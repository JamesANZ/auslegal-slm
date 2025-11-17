#!/usr/bin/env python3
"""
Query Legal SLM

Interactive CLI for querying the fine-tuned legal SLM.
"""

import os
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LegalSLM:
    """Wrapper class for the fine-tuned legal SLM."""
    
    def __init__(self, model_dir: str = "models/legal_slm"):
        """
        Initialize the Legal SLM.
        
        Args:
            model_dir: Path to directory containing the fine-tuned model
        """
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from {model_dir}...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("Model loaded successfully!")
    
    def generate_answer(
        self,
        question: str,
        temperature: float = 0.4,
        max_length: int = 250,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate an answer to a legal question.
        
        Args:
            question: The legal question to answer
            temperature: Sampling temperature (lower = more deterministic)
            max_length: Maximum length of generated response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated answer text
        """
        # Build prompt
        prompt = f"Based on Australian legal documents, answer the following.\n\nQuestion: {question}\nAnswer:"
        
        # Tokenize prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Reduce repetition
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part (after "Answer:")
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            # Fallback: return everything after the prompt
            answer = full_response[len(prompt):].strip()
        
        return answer
    
    def interactive_query(self):
        """Run interactive query loop."""
        print("\n" + "=" * 80)
        print("Legal SLM Query Interface")
        print("=" * 80)
        print("Enter your legal questions (type 'quit' or 'exit' to stop)")
        print()
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nGenerating answer...")
                answer = self.generate_answer(question)
                print(f"\nAnswer: {answer}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query the Legal SLM")
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/legal_slm',
        help='Path to fine-tuned model directory'
    )
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='Single question to ask (if not provided, runs interactive mode)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.4,
        help='Sampling temperature (default: 0.4)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=250,
        help='Maximum response length in tokens (default: 250)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        print("  Please train the model first using train_slm.py")
        sys.exit(1)
    
    # Check if model files exist (support both pytorch_model.bin and model.safetensors)
    config_file = os.path.join(args.model_dir, 'config.json')
    model_file_bin = os.path.join(args.model_dir, 'pytorch_model.bin')
    model_file_safetensors = os.path.join(args.model_dir, 'model.safetensors')
    
    if not os.path.exists(config_file):
        print(f"ERROR: config.json not found in {args.model_dir}")
        print("  Please train the model first using train_slm.py")
        sys.exit(1)
    
    if not os.path.exists(model_file_bin) and not os.path.exists(model_file_safetensors):
        print(f"ERROR: Model file not found in {args.model_dir}")
        print("  Expected either pytorch_model.bin or model.safetensors")
        print("  Please train the model first using train_slm.py")
        sys.exit(1)
    
    # Initialize SLM
    slm = LegalSLM(model_dir=args.model_dir)
    
    # Single question or interactive mode
    if args.question:
        answer = slm.generate_answer(
            args.question,
            temperature=args.temperature,
            max_length=args.max_length
        )
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}\n")
    else:
        slm.interactive_query()


if __name__ == "__main__":
    main()

