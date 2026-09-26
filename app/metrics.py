from prometheus_client import Counter, Gauge, Histogram

requests_total = Counter(
    "llm_requests_total",
    "Total number of requests",
    ["endpoint", "status"],
)

request_duration = Histogram(
    "llm_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

requests_in_progress = Gauge(
    "llm_requests_in_progress",
    "Number of requests currently being processed",
    ["endpoint"],
)

tokens_generated = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated",
)

backend_errors = Counter(
    "llm_backend_errors_total",
    "Errors from the LLM backend",
    ["error_type"],
)
