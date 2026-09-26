import json
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import settings
from app.llm import LLMBackendError, generate, generate_stream, list_models
from app.logging_config import setup_logging
from app.metrics import (
    request_duration,
    requests_in_progress,
    requests_total,
)
from app.schemas import ChatRequest, ChatResponse, ModelsResponse

# test
app = FastAPI(title="LLM Service", version="0.1.0")

setup_logging(settings.log_level)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        await list_models()
    except LLMBackendError:
        raise HTTPException(status_code=503, detail="LLM backend not ready")
    return {"status": "ready"}


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    endpoint = request.url.path

    if endpoint == "/metrics":
        return await call_next(request)

    requests_in_progress.labels(endpoint=endpoint).inc()
    start = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        duration = time.perf_counter() - start
        requests_in_progress.labels(endpoint=endpoint).dec()
        request_duration.labels(endpoint=endpoint).observe(duration)
        requests_total.labels(endpoint=endpoint, status=status).inc()

    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await generate(
            message=request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except LLMBackendError as exc:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")

    return ChatResponse(**result)


@app.get("/v1/models", response_model=ModelsResponse)
async def models():
    try:
        result = await list_models()
    except LLMBackendError as exc:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")

    return ModelsResponse(models=result)


@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            async for chunk in generate_stream(
                message=request.message,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except LLMBackendError:
            yield 'data: {"type": "error", "message": "LLM backend unavailable"}\n\n'

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
