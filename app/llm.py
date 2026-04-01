"""OpenRouter LLM integration using the chat completions API."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "minimax/minimax-m2.5"

SYSTEM_PROMPT = (
    "You are Consilium, a medical AI assistant for doctors in a hospital setting. "
    "Provide evidence-based, concise medical information. "
    "Note: you normally have access to the hospital's clinical knowledge base "
    "(textbooks, protocols, guidelines). If you are answering without knowledge "
    "base context, explicitly state that your recommendations are based on "
    "general medical knowledge of the model rather than the knowledge base. "
    "For concrete recommendations, make that provenance clear in the wording, "
    "for example by noting that a point is a general clinical consideration "
    "based on model knowledge and not a knowledge-base citation. "
    "Always clarify that your responses are for informational purposes only and "
    "should not replace clinical judgement. "
)


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_CONSILIUM_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_CONSILIUM_KEY environment variable is not set")
    return key


def build_messages(
    history: list[dict[str, str]],
    user_content: str,
) -> list[dict[str, str]]:
    """Build the messages payload for the OpenRouter API.

    *history* should be a list of ``{"role": "user"|"assistant", "content": "..."}``
    dicts representing the prior conversation turns.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def chat(
    history: list[dict[str, str]],
    user_content: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request to OpenRouter and return the assistant reply."""
    api_key = _get_api_key()

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": build_messages(history, user_content),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://consilium.kz",
        "X-Title": "Consilium",
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OPENROUTER_URL, json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def chat_stream(
    history: list[dict[str, str]],
    user_content: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Stream chat completion tokens from OpenRouter.

    Yields successive content-delta strings as they arrive.
    """
    api_key = _get_api_key()

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": build_messages(history, user_content),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://consilium.kz",
        "X-Title": "Consilium",
    }

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
