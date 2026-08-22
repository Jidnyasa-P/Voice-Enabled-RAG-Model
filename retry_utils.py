"""
Person 3 — Generic retry wrapper.
Used by pipeline.py around every external call (STT, guardrail LLM calls,
retrieval, generation) so a single transient failure doesn't crash the
whole pipeline.
"""

import time


def with_retry(fn, *args, retries: int = 2, backoff: float = 0.5, **kwargs):
    """
    Calls fn(*args, **kwargs), retrying on any exception up to `retries`
    additional times with linear backoff. Re-raises the last exception if
    every attempt fails.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt == retries:
                raise
            time.sleep(backoff * (attempt + 1))
    raise last_error  # unreachable, but keeps type-checkers happy
