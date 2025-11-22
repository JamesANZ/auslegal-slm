#!/usr/bin/env python3
"""
Legal RAG System

Core RAG system using LangChain for Australian legal documents.
Supports multiple LLM providers from cross-llm-mcp.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Add parent directory to path to import clean_legal_text
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI as OpenAIChat
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_mistralai import ChatMistralAI
    from langchain_core.language_models import BaseChatModel

    # Try langchain_huggingface first (newer), fallback to langchain_community
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    # Try langchain_classic first (LangChain 1.0+)
    try:
        from langchain_classic.chains import RetrievalQAWithSourcesChain
    except ImportError:
        # Fallback to langchain.chains for older versions
        from langchain.chains import RetrievalQAWithSourcesChain
except ImportError:
    # Fallback for older LangChain versions
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI as OpenAIChat
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_mistralai import ChatMistralAI
    from langchain_core.language_models import BaseChatModel

    # Try langchain_classic first
    try:
        from langchain_classic.chains import RetrievalQAWithSourcesChain
    except ImportError:
        from langchain.chains import RetrievalQAWithSourcesChain

from rag.config import (
    LLM_PROVIDERS,
    SUPPORTED_LLMS,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTORSTORE_PATH,
    DEFAULT_TOP_K,
    validate_api_key,
    get_api_key,
)


# Import cleaning function from parent directory
try:
    from clean_data import clean_legal_text
except ImportError:
    # Fallback if import fails
    import re

    def clean_legal_text(text: str) -> str:
        """Fallback cleaning function."""
        lines = text.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("=" * 20):
                start_idx = i + 1
                break
            if line.startswith("URL:") or line.startswith("Scraped:"):
                start_idx = i + 1

        content_lines = lines[start_idx:] if start_idx > 0 else lines
        filtered_lines = []
        for line in content_lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            filtered_lines.append(line)

        cleaned_text = "\n".join(filtered_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        return cleaned_text.strip()


class CustomTextLoader(TextLoader):
    """Custom text loader that applies cleaning to legal documents."""

    def load(self) -> List:
        """Load and clean documents."""
        docs = super().load()
        for doc in docs:
            doc.page_content = clean_legal_text(doc.page_content)
        return docs


class LegalRAG:
    """RAG system for Australian legal documents."""

    def __init__(
        self,
        vectorstore_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """
        Initialize Legal RAG system.

        Args:
            vectorstore_path: Path to vector store directory
            data_dir: Path to data directory with legal documents
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.vectorstore_path = vectorstore_path or VECTORSTORE_PATH
        self.data_dir = data_dir or "data"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings (local, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        self.vectorstore = None
        self.qa_chain = None

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build or load the vector store index.

        Args:
            force_rebuild: If True, rebuild index even if it exists
        """
        vectorstore_dir = Path(self.vectorstore_path)

        # Check if index exists
        if (
            not force_rebuild
            and vectorstore_dir.exists()
            and list(vectorstore_dir.glob("*.faiss"))
        ):
            print(f"Loading existing vector store from {self.vectorstore_path}...")
            try:
                # Try with allow_dangerous_deserialization parameter (newer versions)
                try:
                    self.vectorstore = FAISS.load_local(
                        str(vectorstore_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except TypeError:
                    # Fallback for older versions without the parameter
                    self.vectorstore = FAISS.load_local(
                        str(vectorstore_dir), self.embeddings
                    )
                print(
                    f"Loaded {self.vectorstore.index.ntotal} documents from vector store."
                )
                return
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Rebuilding index...")

        # Build new index
        print(f"Building vector store index from {self.data_dir}...")

        # Load documents
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        # Use custom loader with cleaning
        loader = DirectoryLoader(
            str(data_path),
            glob="**/*.txt",
            loader_cls=CustomTextLoader,
            show_progress=True,
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")

        # Split documents into chunks
        print(
            f"Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})..."
        )
        texts = self.text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks")

        # Create vector store
        print("Generating embeddings and creating vector store...")
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)

        # Save vector store
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(vectorstore_dir))
        print(f"Vector store saved to {self.vectorstore_path}")
        print(f"Index contains {self.vectorstore.index.ntotal} document chunks")

    def _create_llm(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> BaseChatModel:
        """
        Create LLM instance for the specified provider.

        Args:
            provider: LLM provider name
            model: Model name (uses default if not specified)
            api_key: API key (uses env var if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LangChain LLM instance

        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported: {', '.join(SUPPORTED_LLMS)}"
            )

        config = LLM_PROVIDERS[provider]
        api_key = validate_api_key(provider, api_key)
        model = model or config.default_model

        # Create LLM based on provider
        if provider == "chatgpt":
            return OpenAIChat(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "claude":
            return ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        elif provider == "mistral":
            return ChatMistralAI(
                model=model,
                mistral_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider in ["deepseek", "grok", "kimi", "perplexity"]:
            # These use OpenAI-compatible API with custom base URL
            # Use the base_url from config (already set correctly)
            base_url = config.base_url

            return OpenAIChat(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        else:
            raise ValueError(f"LLM creation not implemented for provider: {provider}")

    def setup_qa_chain(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        Set up the QA chain with the specified LLM.

        Args:
            provider: LLM provider name
            model: Model name
            api_key: API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_k: Number of documents to retrieve
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not loaded. Call build_index() first.")

        # Create LLM
        llm = self._create_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Create prompt template
        prompt_template = """Use the following pieces of context from Australian legal documents to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context from documents:
{summaries}

Question: {question}

Answer based on the context provided:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["summaries", "question"]
        )

        # Create QA chain with sources
        try:
            # Try newer API with return_source_documents
            self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
            )
        except TypeError:
            # Fallback for older versions
            self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
            )

    def query(
        self,
        question: str,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question
            provider: LLM provider name
            model: Model name
            api_key: API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not loaded. Call build_index() first.")

        # Set up QA chain if not already set up or if parameters changed
        self.setup_qa_chain(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
        )

        # Run query
        result = self.qa_chain.invoke({"question": question})

        # Extract source documents
        source_docs = result.get("source_documents", [])
        if not source_docs:
            # Try alternative key
            source_docs = result.get("docs", [])

        sources = []
        for doc in source_docs:
            sources.append(
                {
                    "content": (
                        doc.page_content if hasattr(doc, "page_content") else str(doc)
                    ),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                }
            )

        # Get actual model used from the LLM response
        requested_model = model or LLM_PROVIDERS[provider].default_model
        actual_model = requested_model

        # Try to get actual model from the query result's response metadata
        # Check if result has llm_output with model_name
        if (
            hasattr(result, "llm_output")
            and result.llm_output
            and isinstance(result.llm_output, dict)
        ):
            if "model_name" in result.llm_output:
                actual_model = result.llm_output["model_name"]

        # Also check the LLM chain's last response if available
        # Try to get from the chain's internal state without making extra API calls
        if (
            self.qa_chain
            and hasattr(self.qa_chain, "llm_chain")
            and hasattr(self.qa_chain.llm_chain, "llm")
        ):
            llm = self.qa_chain.llm_chain.llm
            # For OpenAI-compatible models, the model name is usually the requested model
            # unless OpenAI resolves it to a specific snapshot (e.g., gpt-4 -> gpt-4-0613)
            # We'll use the requested model unless we can get the actual from the result
            if hasattr(llm, "model_name") and llm.model_name:
                # This is the requested model, not necessarily the actual one
                pass

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", ""),
            "source_documents": sources,
            "provider": provider,
            "model": actual_model,
            "model_requested": model or LLM_PROVIDERS[provider].default_model,
        }
