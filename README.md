# llm-service

Production-shaped HTTP service around a local LLM. FastAPI in front, Ollama
behind it, everything in containers.

The point of the project is not "call a model from Python" — it is the layer
around that call: validation, streaming, typed errors, configuration, metrics,
health checks, tests and CI.

---

## Why a service instead of calling Ollama directly

Ollama already exposes an HTTP API, so the wrapper has to earn its place. It does
four things Ollama does not:

- **Validation and business logic** — request limits, prompt handling, model
  selection per task
- **Error contract** — a caller gets `502 Model backend is unavailable`, never an
  internal URL or a stack trace
- **Observability** — structured logs and Prometheus metrics for every request
- **Isolation** — the API is a contract. Swapping Ollama for vLLM changes one
  module (`app/llm.py`); the endpoints, schemas and clients stay untouched

---

## Endpoints

| method | path | purpose |
|---|---|---|
| `GET` | `/health` | liveness — is the process alive |
| `GET` | `/ready` | readiness — is the backend reachable (`503` if not) |
| `POST` | `/v1/chat` | full response in one payload |
| `POST` | `/v1/chat/stream` | token-by-token stream over SSE |
| `GET` | `/v1/models` | available models, in this service's own format |
| `GET` | `/metrics` | Prometheus exposition format |

Interactive docs are generated from the type annotations and live at `/docs`.

### Example

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain embeddings in one sentence.", "temperature": 0}'
```

```json
{
  "reply": "Embeddings are dense numeric vectors that place semantically similar text close together in space.",
  "model": "qwen2.5:0.5b-instruct-q8_0",
  "prompt_tokens": 38,
  "completion_tokens": 24,
  "latency_ms": 612.4
}
```

Streaming (note `-N`, otherwise curl buffers the whole response):

```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Count from one to ten."}'
```

```
data: {"type": "token", "text": "One"}

data: {"type": "token", "text": ", two"}

data: {"type": "done", "prompt_tokens": 34, "completion_tokens": 29}

data: [DONE]
```

---

## Running it

```bash
git clone https://github.com/<user>/llm-service.git
cd llm-service
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:0.5b-instruct-q8_0
curl http://localhost:8000/ready
```

That is the whole setup. No Python install, no virtualenv, no Ollama on the host.

### Development mode

`docker-compose.override.yml` is merged automatically and adds a bind mount plus
`--reload`, so edits to `app/` restart the service without rebuilding the image.

### Configuration

Everything is read from the environment, with defaults in `app/config.py`:

| variable | default | meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | backend address |
| `MODEL` | `qwen2.5:0.5b-instruct-q8_0` | model tag |
| `REQUEST_TIMEOUT` | `120` | seconds before giving up on the backend |
| `LOG_LEVEL` | `INFO` | logging verbosity |

The same image runs locally, in compose and in production — only the environment
changes.

---

## Design decisions

### Streaming exists for perceived latency, not throughput

On the reference CPU setup a 200-token answer takes ~4.4 s at 45 tok/s, while the
first token is ready after ~120 ms. Without streaming the user stares at nothing
for four seconds; with it, text starts flowing immediately. Total time is
identical.

Server-Sent Events rather than WebSocket: the stream is one-directional, runs over
plain HTTP and needs no special client. The response sets
`X-Accel-Buffering: no`, because a proxy that buffers the stream silently undoes
the entire feature.

One consequence worth knowing: **inside a stream, errors have to travel in the
payload**. The status code and headers are already on the wire by the time
generation starts, so raising an HTTP error is no longer possible — a failure
mid-stream is emitted as `{"type": "error", ...}`.

### Errors are typed, and the client sees none of the internals

`httpx` failures are split into three cases, because they call for different
reactions:

| exception | meaning | logged | returned to client |
|---|---|---|---|
| `TimeoutException` | backend alive, too slow | timeout value, exception | `Model did not respond in time` |
| `HTTPStatusError` | backend answered with an error code | status + first 200 chars of body | `Model backend returned an error` |
| `RequestError` | backend unreachable | full URL and cause | `Model backend is unavailable` |

All three are re-raised as a custom `LLMBackendError`, which the endpoint turns
into `502`. That matters: a bare `except Exception` would have swallowed bugs in
this service's own code and reported them as backend failures. Now `502` means
"the dependency is at fault" and `500` means "I am", and the two never get
confused.

A regression test asserts that no internal address ever appears in a client-facing
error — an early version leaked `localhost:11434` in the error detail.

### Metrics

| metric | type | labels |
|---|---|---|
| `llm_requests_total` | counter | `endpoint`, `status` |
| `llm_request_duration_seconds` | histogram | `endpoint` |
| `llm_requests_in_progress` | gauge | `endpoint` |
| `llm_tokens_generated_total` | counter | — |
| `llm_backend_errors_total` | counter | `error_type` |

Latency is a histogram rather than an average on purpose: 99 requests at 0.1 s
plus one at 30 s average to a reassuring 0.4 s, while one user waited half a
minute. Buckets are chosen around measured CPU inference times.

Collection happens in middleware, so no endpoint has to remember to instrument
itself. The counters are updated inside `finally` — without it the in-progress
gauge never decrements on an exception and climbs forever.

Label cardinality is kept deliberately low. Anything unbounded (user id, prompt
text) as a label would create a time series per value and take Prometheus down.

### Liveness and readiness are separate

`/health` reports whether the process is alive; `/ready` actually calls the
backend. The distinction is operational: Ollama may spend a minute loading a
model, during which the service is perfectly healthy but cannot serve traffic.
Restarting it — which is what a failing liveness probe triggers — would not help.

The same idea is applied at the infrastructure level: `depends_on` in compose uses
`condition: service_healthy`, because plain `depends_on` only waits for the
container to start, not for the service inside it to be usable.

### Image layer order

```dockerfile
COPY requirements.txt .                  # changes rarely
RUN pip install -r requirements.txt      # slow, cached
COPY app/ ./app/                         # changes constantly
```

A layer's hash includes the hash of the layer beneath it, so changing an early
layer invalidates every layer above it. With the order reversed, every single code
edit reinstalls all dependencies. Rule: most stable first.

`--host 0.0.0.0` in the entrypoint is not optional — the default `127.0.0.1` means
"this machine", and inside a container that machine is the container itself.

---

## Testing

```bash
pytest -v
```

`TestClient` calls the application directly, without a network or a running
server, so the suite finishes in seconds. Backend failures are simulated with
`monkeypatch` rather than by stopping Ollama: the tests verify this service's
behaviour, not Ollama's.

Covered: health, metrics exposition, five invalid-payload cases via
`parametrize`, `502` handling with no internal detail leakage.

Deliberately not covered here: tests that hit a real model. They are slow and
depend on an external service, and a suite that takes minutes stops being run.

---

## CI

GitHub Actions on every push to `main` and every pull request:

```
lint (ruff) → format check → tests → docker build
```

The build job has `needs: test`, so no image is produced from a failing commit.

Everything runs on a clean runner, which is the point — "works on my machine" does
not pass. The most common failure mode caught by this setup is an incomplete
`requirements.txt`.

---

## Layout

```
app/
  main.py             endpoints, middleware, app wiring
  schemas.py          pydantic request/response models
  llm.py              Ollama client — knows nothing about FastAPI
  config.py           settings from environment variables
  metrics.py          Prometheus metric definitions
  logging_config.py   logging setup (stdout, for container log collection)
tests/
  test_api.py
Dockerfile
docker-compose.yml
docker-compose.override.yml   local dev: bind mount + --reload
.github/workflows/ci.yml
```

Dependencies point one way: `main.py → llm.py`, `main.py → schemas.py`. `llm.py`
imports nothing from `main.py` and does not import FastAPI at all, which is what
makes it callable from a script or a test. A reverse arrow appearing would be a
sign the structure has drifted.

---

## Related

[llm-playground](https://github.com/<user>/llm-playground) — inference
benchmarks behind the engine and quantization choices used here: transformers vs
Ollama vs vLLM on CPU and GPU, prefill/decode characteristics, KV-cache memory
planning.