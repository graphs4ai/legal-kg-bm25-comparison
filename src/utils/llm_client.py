"""LLM provider clients: Ollama, DeepInfra and Maritaca."""

import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_API_URL = os.getenv("DEEPINFRA_API_URL")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
MARITACA_API_KEY = os.getenv("MARITACA_API_KEY")
MARITACA_API_URL = os.getenv("MARITACA_API_URL", "https://chat.maritaca.ai/api")

# Maritaca BRL per 1M tokens (input, output), https://docs.maritaca.ai/pt/precos;
# night rate 22:00-06:00 BRT.
MARITACA_PRICES_BRL = {
    "sabiazinho-4": {"standard": (1.00, 4.00), "night": (0.70, 2.80)},
    "sabia-4": {"standard": (5.00, 20.00), "night": (3.50, 14.00)},
}
_BRT = ZoneInfo("America/Sao_Paulo")

# One shared client per (endpoint, timeout, max_retries); each holds a connection pool.
_openai_clients: dict[
    tuple[str, float | None, int | None], OpenAI] = {}
_openai_clients_lock = threading.Lock()


def _get_openai_client(
    base_url: str, api_key: str, timeout: float | None,
    max_retries: int | None,
) -> OpenAI:
    """Return the shared OpenAI client for this (endpoint, timeout, retries)."""

    key = (base_url, timeout, max_retries)
    with _openai_clients_lock:
        client = _openai_clients.get(key)
        if client is None:
            client_options = {}
            if timeout is not None:
                client_options["timeout"] = timeout
            if max_retries is not None:
                client_options["max_retries"] = max_retries
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                **client_options,
            )
            _openai_clients[key] = client
        return client


def _get_deepinfra_client(
    timeout: float | None, max_retries: int | None
) -> OpenAI:
    """Return the shared DeepInfra client for this (timeout, max_retries) pair."""

    return _get_openai_client(
        DEEPINFRA_API_URL, DEEPINFRA_API_KEY, timeout, max_retries)


def call_ollama(
    prompt: str,
    model: str,
    temperature: float = 0.5,
    think: bool | str = False,
    num_predict: int = 512,
    format: str | None = None,
    num_ctx: int | None = None,
) -> dict:
    """Call the Ollama API and return the response.

    Args:
        format (str | None): Output format constraint (e.g. "json").
        num_ctx (int | None): Context window; Ollama silently drops the start
            of a longer prompt, so it must cover prompt + num_predict.

    Returns:
        dict: provider, raw_response, text, thinking, input/output/total
            tokens, elapsed_seconds, estimated_cost, finish_reason.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx
    if format is not None:
        payload["format"] = format

    start = time.time()
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    elapsed = time.time() - start
    data = response.json()

    return {
        "provider": "ollama",
        "raw_response": data,
        "text": data.get("response", ""),
        "thinking": data.get("thinking"),

        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),

        "elapsed_seconds": elapsed,
        "estimated_cost": getattr(
            data.get("usage", {}),
            "estimated_cost",
            None,
        ),
        # "stop" when the model finished, "length" when it hit num_predict.
        "finish_reason": data.get("done_reason"),
    }


def call_deepinfra(
    prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Call a DeepInfra-hosted model through the OpenAI-compatible API.

    Args:
        response_format (dict | None): Structured-output constraint, e.g.
            {"type": "json_object"}.
        timeout (float | None): Per-request deadline in seconds; None uses
            the client default (600s).
        max_retries (int | None): Retries on timeout and connection errors;
            None uses the client default (2).
        chat_template_kwargs (dict | None): Chat-template variables, e.g.
            {"enable_thinking": False}.

    Returns:
        dict: Same keys as ``call_ollama``.
    """

    if not DEEPINFRA_API_KEY:
        raise ValueError(
            "DEEPINFRA_API_KEY environment variable is not set."
        )

    client = _get_deepinfra_client(timeout, max_retries)

    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    start = time.time()
    extra = {}
    if response_format is not None:
        extra["response_format"] = response_format
    if chat_template_kwargs is not None:
        extra["extra_body"] = {"chat_template_kwargs": chat_template_kwargs}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        **extra,
    )
    elapsed = time.time() - start

    choice = response.choices[0]
    usage = response.usage

    return {
        "provider": "deepinfra",
        "raw_response": response.model_dump(),
        "text": choice.message.content or "",
        "thinking": getattr(
            choice.message,
            "reasoning_content",
            None,
        ),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "elapsed_seconds": elapsed,
        "estimated_cost": getattr(
            usage,
            "estimated_cost",
            None,
        ),
        "finish_reason": choice.finish_reason,
    }


def maritaca_rate(at: datetime | None = None) -> str:
    """Maritaca price tier at an aware moment (default now): "night" 22:00-06:00 BRT, else "standard"."""

    hour = (at or datetime.now(_BRT)).astimezone(_BRT).hour
    return "night" if hour >= 22 or hour < 6 else "standard"


def maritaca_cost_brl(
    model: str, input_tokens: int, output_tokens: int, rate: str
) -> float | None:
    """Cost of one Maritaca call in BRL at a "standard" or "night" rate; None for an unpriced model."""

    prices = MARITACA_PRICES_BRL.get(model)
    if prices is None:
        return None
    price_in, price_out = prices[rate]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def call_maritaca(
    prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    system_prompt: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = 8,
) -> dict:
    """Call a Maritaca (Sabiá) model through its OpenAI-compatible API.

    The API reports no cost, so ``estimated_cost`` is computed in BRL at the
    tier in force when the request starts. ``max_retries`` is high because the
    entry tier allows 60 requests per minute.

    Returns:
        dict: Same keys as ``call_ollama``, plus ``cost_currency`` ("BRL")
            and ``cost_rate``.
    """

    if not MARITACA_API_KEY:
        raise ValueError("MARITACA_API_KEY environment variable is not set.")

    client = _get_openai_client(
        MARITACA_API_URL, MARITACA_API_KEY, timeout, max_retries)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    rate = maritaca_rate()
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start

    choice = response.choices[0]
    usage = response.usage

    return {
        "provider": "maritaca",
        "raw_response": response.model_dump(),
        "text": choice.message.content or "",
        "thinking": getattr(choice.message, "reasoning_content", None),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "elapsed_seconds": elapsed,
        "estimated_cost": maritaca_cost_brl(
            model, usage.prompt_tokens, usage.completion_tokens, rate),
        "cost_currency": "BRL",
        "cost_rate": rate,
        "finish_reason": choice.finish_reason,
    }
