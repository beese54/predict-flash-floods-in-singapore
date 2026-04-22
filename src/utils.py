"""
Centralised API client factory and config loader.
All API keys are loaded from .env — never hardcoded anywhere else.
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file: src/ -> project root)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_config() -> dict:
    """Load project config.yaml."""
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def get_openai_client():
    """Return an authenticated OpenAI client. Raises if key not set."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return OpenAI(api_key=api_key)


def get_anthropic_client():
    """Return an authenticated Anthropic client. Raises if key not set."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("sk-ant-..."):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return Anthropic(api_key=api_key)


def get_llm_client():
    """Return the LLM client configured in config.yaml (openai or anthropic)."""
    config = get_config()
    provider = config["llm"]["provider"]
    if provider == "openai":
        return get_openai_client(), "openai"
    elif provider == "anthropic":
        return get_anthropic_client(), "anthropic"
    else:
        raise ValueError(f"Unknown LLM provider in config.yaml: {provider!r}")


def project_root() -> Path:
    return _PROJECT_ROOT
