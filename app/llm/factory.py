"""Builds the gateway for the configured provider.

This is the one place that knows which provider exists. Adding a provider (or,
later, BYOK credential injection) happens here.
"""

from __future__ import annotations

import json

from app.core.config import Settings
from app.llm.gateway import LLMGateway


# Per-provider default models, applied when LLM_MODEL still holds another
# provider's name (sending "claude-opus-5" to OpenAI would 404).
PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    # gpt-oss-120b: the strongest chat model the user's Groq key can access
    # (llama-3.3-70b returns model_not_found on it, checked 2026-08-24).
    "groq": "openai/gpt-oss-120b",
}

# Model-name prefixes that identify a provider's own models. Groq hosts many
# open-weight families, so it gets a tuple; "openai/gpt-oss-*" is Groq's id
# for OpenAI's open-weight models, not an OpenAI-API model.
_PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude",),
    "openai": ("gpt",),
    "groq": ("llama", "meta-llama/", "openai/", "gemma", "qwen", "deepseek", "moonshotai/"),
}


def resolve_model(provider: str, configured: str) -> str:
    prefixes = _PROVIDER_PREFIXES.get(provider)
    if prefixes and not configured.startswith(prefixes):
        return PROVIDER_DEFAULT_MODEL.get(provider, configured)
    return configured


def build_gateway(settings: Settings) -> LLMGateway:
    if settings.llm_provider in ("anthropic", "openai", "groq"):
        from app.llm.langchain_gateway import LangChainGateway, build_chat_model

        model = resolve_model(settings.llm_provider, settings.llm_model)
        api_key = {
            "anthropic": settings.anthropic_api_key,
            "openai": settings.openai_api_key,
            "groq": settings.groq_api_key,
        }[settings.llm_provider]
        return LangChainGateway(
            build_chat_model(
                provider=settings.llm_provider,
                model=model,
                api_key=api_key,
                timeout_seconds=settings.llm_timeout_seconds,
            ),
            model_name=model,
            structured_attempts=settings.llm_structured_attempts,
        )

    if settings.llm_provider == "fake":
        from app.llm.fake import FakeLLMGateway, RecordedCall

        def canned(call: RecordedCall) -> str:
            if call.purpose == "ingest_resume":
                return json.dumps(
                    {
                        "claims": [
                            {
                                "text": "Built a RAG system using FAISS and FastAPI.",
                                "skills": ["RAG", "FAISS", "FastAPI"],
                                "category": "project",
                            }
                        ],
                        "focus_areas": ["RAG"],
                        "seniority_signal": "mid-level",
                    }
                )
            if call.purpose == "ingest_job_description":
                return json.dumps(
                    {
                        "role_title": "AI Engineer",
                        "requirements": [
                            {
                                "skill": "RAG",
                                "importance": "must_have",
                                "evidence": "stubbed job description",
                            }
                        ],
                    }
                )
            if call.purpose == "evaluate_answer":
                return json.dumps(
                    {
                        "correctness": 6.0,
                        "depth": 5.0,
                        "communication": 7.0,
                        "dimension_scores": [],
                        "concepts_covered": ["stub concept"],
                        "missing_concepts": ["stub gap"],
                        "misconceptions": [],
                        "recommended_action": "probe_deeper",
                        "follow_up_topic": None,
                        "rationale": "Stubbed evaluation from the fake provider.",
                    }
                )
            return json.dumps(
                {
                    "question": "Stubbed question from the fake provider. Walk me through your reasoning.",
                    "topic": "stub topic",
                }
            )

        return FakeLLMGateway(
            responder=canned, structured_attempts=settings.llm_structured_attempts
        )

    raise ValueError(f"unsupported llm_provider: {settings.llm_provider}")
