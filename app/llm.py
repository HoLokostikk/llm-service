import json
import logging
import time

import httpx

from app.config import settings
from app.metrics import backend_errors, tokens_generated

logger = logging.getLogger(__name__)


class LLMBackendError(Exception):
    ""


async def generate(message: str, temperature: float, max_tokens: int) -> dict:
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.model,
                    "prompt": message,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as exc:
        backend_errors.labels(error_type="timeout").inc()
        logger.error("Ollama timeout after %ss: %s", settings.request_timeout, exc)
        raise LLMBackendError("Model did not respond in time") from exc

    except httpx.HTTPStatusError as exc:
        backend_errors.labels(error_type="http_status").inc()
        logger.error(
            "Ollama returned %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        raise LLMBackendError("Model backend returned an error") from exc

    except httpx.RequestError as exc:
        backend_errors.labels(error_type="connection").inc()
        logger.error("Cannot reach Ollama at %s: %s", settings.ollama_url, exc)
        raise LLMBackendError("Model backend is unavailable") from exc

    completion_tokens = data.get("eval_count", 0)
    latency_ms = 1000 * (time.perf_counter() - start)

    tokens_generated.inc(completion_tokens)
    logger.info("generated tokens=%s latency=%.0fms", completion_tokens, latency_ms)

    return {
        "reply": data["response"].strip(),
        "model": settings.model,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
    }


async def generate_stream(message: str, temperature: float, max_tokens: int):
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.model,
                    "prompt": message,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    chunk = json.loads(line)

                    if chunk.get("done"):
                        completion_tokens = chunk.get("eval_count", 0)
                        tokens_generated.inc(completion_tokens)
                        yield {
                            "type": "done",
                            "prompt_tokens": chunk.get("prompt_eval_count", 0),
                            "completion_tokens": completion_tokens,
                        }
                    else:
                        yield {"type": "token", "text": chunk.get("response", "")}

    except httpx.TimeoutException as exc:
        backend_errors.labels(error_type="timeout").inc()
        logger.error("Ollama timeout during stream: %s", exc)
        raise LLMBackendError("Model did not respond in time") from exc

    except httpx.HTTPStatusError as exc:
        backend_errors.labels(error_type="http_status").inc()
        logger.error("Ollama returned %s during stream", exc.response.status_code)
        raise LLMBackendError("Model backend returned an error") from exc

    except httpx.RequestError as exc:
        backend_errors.labels(error_type="connection").inc()
        logger.error("Cannot reach Ollama at %s: %s", settings.ollama_url, exc)
        raise LLMBackendError("Model backend is unavailable") from exc


async def list_models() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()

    except httpx.HTTPStatusError as exc:
        backend_errors.labels(error_type="http_status").inc()
        logger.error("Ollama returned %s on /api/tags", exc.response.status_code)
        raise LLMBackendError("Model backend returned an error") from exc

    except httpx.RequestError as exc:
        backend_errors.labels(error_type="connection").inc()
        logger.error("Cannot reach Ollama at %s: %s", settings.ollama_url, exc)
        raise LLMBackendError("Model backend is unavailable") from exc

    return [
        {"name": m["name"], "size_gb": m["size"] / 1e9} for m in data.get("models", [])
    ]
