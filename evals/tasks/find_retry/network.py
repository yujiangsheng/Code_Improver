"""Network helpers."""
import time


def sleep_and_redo(operation):
    """Run *operation* up to 5 times, doubling the wait between attempts."""
    last_exc = None
    for i in range(5):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            time.sleep(2 ** i)
    raise last_exc
