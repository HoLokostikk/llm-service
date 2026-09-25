from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from app.llm import generate, list_models, generate_stream, LLMBackendError
from app.schemas import ChatRequest, ChatResponse, ModelsResponse
from app.logging_config import setup_logging

app = FastAPI(title="LLM Service", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await generate(
            message=request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")

    return ChatResponse(**result)

@app.get("/v1/models", response_model=ModelsResponse)
async def models():
    try:
        result = await list_models()
    except Exception as exc:
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
        except Exception:
            yield f'data: {{"type": "error", "message": "LLM backend unavailable"}}\n\n'

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

