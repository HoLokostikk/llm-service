from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message : str = Field(..., min_length = 1, max_length = 4000)
    temperature : float = Field(0.7, ge = 0.0 , le = 2.0)
    max_tokens : int = Field(512, ge= 1, le = 2048)

class ChatResponse(BaseModel):
    reply : str
    model : str
    prompt_tokens : int
    completion_tokens : int
    latency_ms : float

class ModelInfo(BaseModel):
    name: str
    size_gb: float


class ModelsResponse(BaseModel):
    models: list[ModelInfo]

