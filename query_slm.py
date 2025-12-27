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
from enum import Enum


class PromptStrategy(Enum):
    """Prompting strategies optimized for SLMs."""
    FEW_SHOT = "few_shot"  # Show examples of desired format
    DIRECT = "direct"  # Simple direct question
    CONTINUATION = "continuation"  # Frame as text continuation (matches training)
    STRUCTURED = "structured"  # Highly structured with delimiters


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

    def _build_prompt(
        self, question: str, strategy: PromptStrategy = PromptStrategy.FEW_SHOT
    ) -> str:
        """
        Build prompt using specified strategy.

        For SLMs, few-shot examples work best because they show the exact format
        the model should follow, rather than relying on instruction-following.
        """
        sanitized_question = self._sanitize_input(question)

        if strategy == PromptStrategy.FEW_SHOT:
            # Few-shot: Show examples of Q&A format
            # This works better for SLMs than instructions
            prompt = f"""Question: What is negligence in Australian law?
Answer: Negligence is a tort in Australian law requiring a duty of care, breach of that duty, causation, and damage.

Question: What is the standard of proof in civil cases?
Answer: The standard of proof in Australian civil cases is the balance of probabilities.

Question: {sanitized_question}
Answer:"""
        
        elif strategy == PromptStrategy.DIRECT:
            # Simple direct format (original approach)
            prompt = f"Question: {sanitized_question}\nAnswer:"
        
        elif strategy == PromptStrategy.CONTINUATION:
            # Frame as text continuation (matches training data better)
            # Since model was trained on raw legal documents, this works best
            # Convert question to legal definition style
            question_lower = sanitized_question.lower()
            # Remove question words and convert to definition prompt
            if question_lower.startswith("what is"):
                term = question_lower.replace("what is", "").replace("?", "").strip()
                prompt = f"In Australian law, {term} is defined as"
            elif question_lower.startswith("what are"):
                term = question_lower.replace("what are", "").replace("?", "").strip()
                prompt = f"In Australian law, {term} are defined as"
            elif question_lower.startswith("what does"):
                term = question_lower.replace("what does", "").replace("mean", "").replace("?", "").strip()
                prompt = f"In Australian law, {term} means"
            else:
                # Fallback to general continuation
                prompt = f"In Australian law, regarding {question_lower}, the relevant legal principles are:"
        
        elif strategy == PromptStrategy.STRUCTURED:
            # Highly structured with clear delimiters
            prompt = f"""<QUESTION>
{sanitized_question}
</QUESTION>
<ANSWER>"""
        
        else:
            prompt = f"Question: {sanitized_question}\nAnswer:"

        return prompt

    def generate_answer(
        self,
        question: str,
        temperature: float = 0.2,  # Lower default for more deterministic outputs
        max_length: int = 300,  # Increased for complete answers
        top_p: float = 0.9,
        top_k: int = 40,  # Lower for more focused generation
        use_greedy: bool = False,  # Greedy decoding for exact answers
        num_beams: int = 1,  # Beam search (1 = greedy, >1 = beam search)
        strategy: PromptStrategy = PromptStrategy.FEW_SHOT,
        stop_sequences: list = None,
    ) -> str:
        """
        Generate an answer to a legal question with improved prompting for SLMs.

        Args:
            question: The legal question to answer
            temperature: Sampling temperature (lower = more deterministic)
                         Use 0.0 with use_greedy=True for most deterministic
            max_length: Maximum length of generated response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            use_greedy: If True, use greedy decoding (temperature ignored)
            num_beams: Number of beams for beam search (1 = greedy, >1 = beam search)
            strategy: Prompting strategy to use
            stop_sequences: List of strings to stop generation at (e.g., ["\n\n", "Question:"])

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
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")

        # Build prompt using selected strategy
        prompt = self._build_prompt(sanitized_question, strategy)

        # Tokenize prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Prepare stop sequence token IDs if provided
        stop_token_ids = None
        if stop_sequences:
            stop_token_ids = []
            for seq in stop_sequences:
                token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
                if token_ids:
                    stop_token_ids.append(token_ids[0])  # Use first token as stop

        # Determine generation method
        use_beam_search = num_beams > 1
        use_sampling = not use_greedy and not use_beam_search
        
        # Fix: if temperature is 0.0, use greedy instead
        if temperature == 0.0 and not use_greedy:
            use_greedy = True
            use_sampling = False

        # Generate
        with torch.no_grad():
            if use_greedy or use_beam_search:
                # Greedy or beam search for more deterministic outputs
                generate_kwargs = {
                    "max_length": inputs.shape[1] + max_length,
                    "num_beams": num_beams,
                    "early_stopping": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.2,
                }
                if stop_token_ids:
                    generate_kwargs["forced_eos_token_id"] = stop_token_ids[0]
            else:
                # Sampling-based generation
                generate_kwargs = {
                    "max_length": inputs.shape[1] + max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.2,
                }
                if stop_token_ids:
                    generate_kwargs["forced_eos_token_id"] = stop_token_ids[0]

            outputs = self.model.generate(inputs, **generate_kwargs)

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part
        # Try different extraction methods based on strategy
        answer = None
        
        if strategy == PromptStrategy.STRUCTURED:
            if "</ANSWER>" in full_response:
                answer = full_response.split("</ANSWER>")[0].split("<ANSWER>")[-1].strip()
            elif "<ANSWER>" in full_response:
                answer = full_response.split("<ANSWER>")[-1].strip()
        
        if not answer:
            # Try to find "Answer:" marker
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                # Fallback: return everything after the prompt
                if len(prompt) <= len(full_response):
                    answer = full_response[len(prompt):].strip()
                else:
                    answer = full_response.strip()

        # Apply stop sequences (post-processing)
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in answer:
                    answer = answer.split(stop_seq)[0].strip()

        # Clean up common artifacts
        answer = re.sub(r"^\s*Question:.*$", "", answer, flags=re.MULTILINE)
        answer = answer.strip()

        return answer

    def interactive_query(self):
        """Run interactive query loop."""
        print("\n" + "=" * 80)
        print("Legal SLM Query Interface")
        print("=" * 80)
        print("Enter your legal questions (type 'quit' or 'exit' to stop)")
        print("\nPrompting strategies:")
        print("  - FEW_SHOT (default): Shows examples of desired format")
        print("  - DIRECT: Simple Q&A format")
        print("  - CONTINUATION: Frames as text continuation")
        print("  - STRUCTURED: Uses XML-like delimiters")
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
                    # Use continuation strategy with greedy decoding for most exact answers
                    answer = self.generate_answer(
                        question,
                        temperature=0.0,  # Most deterministic
                        max_length=300,  # Increased for complete answers
                        use_greedy=True,  # Greedy decoding
                        strategy=PromptStrategy.CONTINUATION,  # Best strategy
                        stop_sequences=["\n\nQuestion:", "\n\n\n"],  # Stop at new questions
                    )
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
        default=0.2,
        help="Sampling temperature (default: 0.2, use 0.0 for greedy)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum response length in tokens (default: 300)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (most deterministic, ignores temperature)",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="Number of beams for beam search (default: 1 = greedy)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["few_shot", "direct", "continuation", "structured"],
        default="few_shot",
        help="Prompting strategy (default: few_shot)",
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

    # Convert strategy string to enum
    strategy_map = {
        "few_shot": PromptStrategy.FEW_SHOT,
        "direct": PromptStrategy.DIRECT,
        "continuation": PromptStrategy.CONTINUATION,
        "structured": PromptStrategy.STRUCTURED,
    }
    strategy = strategy_map[args.strategy]

    # Single question or interactive mode
    if args.question:
        try:
            answer = slm.generate_answer(
                args.question,
                temperature=args.temperature,
                max_length=args.max_length,
                use_greedy=args.greedy,
                num_beams=args.beams,
                strategy=strategy,
                stop_sequences=["\n\nQuestion:", "\n\n\n"],
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
