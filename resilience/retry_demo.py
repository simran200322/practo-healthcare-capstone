"""Exponential-backoff retry demonstration around transient node failure (Task 16)."""
import time
import random
from typing import Dict, Any, Callable

# Explicit configuration parameters as required by the PDF
MAX_ATTEMPTS = 4
INITIAL_INTERVAL = 0.10  # 100 milliseconds
MAX_INTERVAL = 1.50      # 1.5 seconds
BACKOFF_FACTOR = 2.0
JITTER_MAX = 0.05        # Random jitter up to 50ms


class TransientClinicalServiceError(Exception):
    """Simulated transient outage error."""
    pass


# Global state tracking transient attempts
_CALL_COUNTER = 0


def simulate_transient_node(record_id: str) -> Dict[str, Any]:
    """Node simulating a transient failure: fails twice, succeeds on the 3rd attempt."""
    global _CALL_COUNTER
    _CALL_COUNTER += 1
    print(f"  --> Executing Node Attempt #{_CALL_COUNTER}...")

    if _CALL_COUNTER < 3:
        raise TransientClinicalServiceError(
            f"Transient upstream timeout (503 Service Unavailable) on attempt #{_CALL_COUNTER}"
        )

    print(f"  --> SUCCESS: Upstream service responded on attempt #{_CALL_COUNTER}!")
    return {
        "status": "success",
        "record_id": record_id,
        "resolved_attempt": _CALL_COUNTER,
        "message": f"Successfully retrieved appointment record {record_id} after transient recovery."
    }


def execute_with_exponential_backoff(
    func: Callable[..., Any],
    *args,
    max_attempts: int = MAX_ATTEMPTS,
    initial_interval: float = INITIAL_INTERVAL,
    max_interval: float = MAX_INTERVAL,
    backoff_factor: float = BACKOFF_FACTOR,
    jitter_max: float = JITTER_MAX,
    **kwargs
) -> Dict[str, Any]:
    """Execute a function with exponential backoff and randomized jitter."""
    attempt = 1
    interval = initial_interval

    while attempt <= max_attempts:
        try:
            return func(*args, **kwargs)
        except TransientClinicalServiceError as exc:
            if attempt == max_attempts:
                print(f"  [RETRY EXHAUSTED] Max attempts ({max_attempts}) reached. Raising error.")
                raise

            jitter = random.uniform(0.0, jitter_max)
            sleep_duration = min(interval + jitter, max_interval)
            print(f"  [RETRY POLICY] Attempt {attempt} failed: {exc}")
            print(f"                 Backing off for {sleep_duration:.3f}s (base: {interval:.2f}s, jitter: {jitter:.3f}s)...")
            time.sleep(sleep_duration)

            interval = min(interval * backoff_factor, max_interval)
            attempt += 1

    raise RuntimeError("Unexpected loop termination in retry policy")


def run_retry_demonstration():
    global _CALL_COUNTER
    _CALL_COUNTER = 0

    print("=" * 75)
    print("PRACTO HEALTHCARE CAPSTONE - EXPONENTIAL BACKOFF RETRY DEMO (Task 16)")
    print("=" * 75)
    print("Configured Parameters:")
    print(f"  - Max Attempts    : {MAX_ATTEMPTS}")
    print(f"  - Initial Interval: {INITIAL_INTERVAL}s")
    print(f"  - Max Interval    : {MAX_INTERVAL}s")
    print(f"  - Backoff Multiplier: {BACKOFF_FACTOR}")
    print(f"  - Jitter Range    : 0.0s to {JITTER_MAX}s")
    print("-" * 75)

    result = execute_with_exponential_backoff(
        simulate_transient_node,
        record_id="APP-0001"
    )

    print("-" * 75)
    print(f"FINAL RESULT: {result['message']}")
    print(f"Recovered on attempt #{result['resolved_attempt']} within configured {MAX_ATTEMPTS} attempts.")
    assert result["resolved_attempt"] == 3
    print("=" * 75)
    print("RETRY RECOVERY DEMONSTRATED WITH 100% SUCCESS!")
    print("=" * 75)
    return result


if __name__ == "__main__":
    run_retry_demonstration()
