"""
Configuration for Legal RAG System

Defines LLM provider configurations, API endpoints, default models,
and other system settings.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    env_var: str
    default_model: str
    base_url: Optional[str] = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "


# LLM Provider Configurations
# Matching cross-llm-mcp defaults and API endpoints
LLM_PROVIDERS: Dict[str, LLMProviderConfig] = {
    "chatgpt": LLMProviderConfig(
        name="ChatGPT",
        env_var="OPENAI_API_KEY",
        default_model="gpt-4",
        base_url="https://api.openai.com/v1",
    ),
    "claude": LLMProviderConfig(
        name="Claude",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-3-5-sonnet-20241022",  # Updated to latest available model
        base_url="https://api.anthropic.com/v1",
    ),
    "deepseek": LLMProviderConfig(
        name="DeepSeek",
        env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    ),
    "gemini": LLMProviderConfig(
        name="Gemini",
        env_var="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1",
    ),
    "grok": LLMProviderConfig(
        name="Grok",
        env_var="XAI_API_KEY",
        default_model="grok-3",
        base_url="https://api.x.ai/v1",
    ),
    "kimi": LLMProviderConfig(
        name="Kimi",
        env_var="KIMI_API_KEY",  # Also supports MOONSHOT_API_KEY
        default_model="moonshot-v1-8k",
        base_url="https://api.moonshot.ai/v1",
    ),
    "perplexity": LLMProviderConfig(
        name="Perplexity",
        env_var="PERPLEXITY_API_KEY",
        default_model="sonar-pro",
        base_url="https://api.perplexity.ai",
    ),
    "mistral": LLMProviderConfig(
        name="Mistral",
        env_var="MISTRAL_API_KEY",
        default_model="mistral-large-latest",
        base_url="https://api.mistral.ai/v1",
    ),
}

# Supported LLM provider names
SUPPORTED_LLMS = list(LLM_PROVIDERS.keys())

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store configuration
VECTORSTORE_PATH = "rag/vectorstore"
VECTORSTORE_INDEX_NAME = "legal_documents"

# Retrieval configuration
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# Data directory
DATA_DIR = "data"


def get_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Get API key for a provider.

    Args:
        provider: LLM provider name
        api_key: Explicitly provided API key (takes precedence)

    Returns:
        API key or None if not found
    """
    if api_key:
        return api_key

    if provider not in LLM_PROVIDERS:
        return None

    config = LLM_PROVIDERS[provider]

    # Check primary env var
    key = os.getenv(config.env_var)
    if key:
        return key

    # Special case for Kimi - also check MOONSHOT_API_KEY
    if provider == "kimi":
        key = os.getenv("MOONSHOT_API_KEY")
        if key:
            return key

    return None


def validate_api_key(provider: str, api_key: Optional[str] = None) -> str:
    """
    Validate and return API key for a provider.

    Args:
        provider: LLM provider name
        api_key: Explicitly provided API key

    Returns:
        API key

    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {', '.join(SUPPORTED_LLMS)}"
        )

    key = get_api_key(provider, api_key)
    if not key:
        config = LLM_PROVIDERS[provider]
        raise ValueError(
            f"{config.name} API key not found. "
            f"Please set {config.env_var} environment variable or provide --api-key"
        )

    return key
