import time

def with_retry(fn, *args, retries=2, backoff=0.5, **kwargs):
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(backoff * (attempt + 1))
