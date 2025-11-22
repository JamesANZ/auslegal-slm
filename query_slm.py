#!/usr/bin/env python3
"""
Query Legal SLM

Interactive CLI for querying the fine-tuned legal SLM.
"""

import os
import sys
import re
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

    def _sanitize_input(self, text: str, max_length: int = 1000) -> str:
        """
        Sanitize user input to prevent prompt injection and ensure reasonable length.

        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not text or not isinstance(text, str):
            raise ValueError("Question must be a non-empty string")

        # Remove potentially harmful patterns (basic prompt injection prevention)
        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)

        # Limit length
        text = text[:max_length].strip()

        # Minimum meaningful length check
        if len(text) < 10:
            raise ValueError(
                "Question is too short. Please provide a more detailed question."
            )

        return text

    def generate_answer(
        self,
        question: str,
        temperature: float = 0.4,
        max_length: int = 250,
        top_p: float = 0.9,
        top_k: int = 50,
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

        Raises:
            ValueError: If question is invalid or too short
        """
        # Validate and sanitize input
        sanitized_question = self._sanitize_input(question)

        # Validate generation parameters
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if max_length < 1 or max_length > 1000:
            raise ValueError("max_length must be between 1 and 1000")

        # Build prompt
        prompt = f"Based on Australian legal documents, answer the following.\n\nQuestion: {sanitized_question}\nAnswer:"

        # Tokenize prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
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
            # Safety check: ensure prompt is not longer than response
            if len(prompt) <= len(full_response):
                answer = full_response[len(prompt) :].strip()
            else:
                # If prompt is longer (shouldn't happen, but handle gracefully)
                answer = full_response.strip()

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

                if question.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break

                if not question:
                    continue

                print("\nGenerating answer...")
                try:
                    answer = self.generate_answer(question)
                    print(f"\nAnswer: {answer}\n")
                except ValueError as e:
                    print(f"\nInvalid input: {e}\n")
                    print("Please try again with a valid question.\n")
                except torch.cuda.OutOfMemoryError:
                    print(
                        "\nError: GPU out of memory. Try reducing max_length or restarting.\n"
                    )
                except Exception as e:
                    print(f"\nUnexpected error: {e}\n")
                    print(
                        "Please try again or contact support if the issue persists.\n"
                    )
                finally:
                    print("-" * 80)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Query the Legal SLM")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/legal_slm",
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to ask (if not provided, runs interactive mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (default: 0.4)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=250,
        help="Maximum response length in tokens (default: 250)",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        print("  Please train the model first using train_slm.py")
        sys.exit(1)

    # Check if model files exist (support both pytorch_model.bin and model.safetensors)
    config_file = os.path.join(args.model_dir, "config.json")
    model_file_bin = os.path.join(args.model_dir, "pytorch_model.bin")
    model_file_safetensors = os.path.join(args.model_dir, "model.safetensors")

    if not os.path.exists(config_file):
        print(f"ERROR: config.json not found in {args.model_dir}")
        print("  Please train the model first using train_slm.py")
        sys.exit(1)

    if not os.path.exists(model_file_bin) and not os.path.exists(
        model_file_safetensors
    ):
        print(f"ERROR: Model file not found in {args.model_dir}")
        print("  Expected either pytorch_model.bin or model.safetensors")
        print("  Please train the model first using train_slm.py")
        sys.exit(1)

    # Initialize SLM
    slm = LegalSLM(model_dir=args.model_dir)

    # Single question or interactive mode
    if args.question:
        try:
            answer = slm.generate_answer(
                args.question, temperature=args.temperature, max_length=args.max_length
            )
            print(f"\nQuestion: {args.question}")
            print(f"Answer: {answer}\n")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
    else:
        slm.interactive_query()


if __name__ == "__main__":
    main()
