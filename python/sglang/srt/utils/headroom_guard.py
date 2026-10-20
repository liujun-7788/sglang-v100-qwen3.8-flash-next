"""Device-memory headroom guard for the scheduler's forward path.

On 2026-10-19 a 158 MiB activation allocation failed with only 82 MiB free
while 1.28 GiB sat in PyTorch's caching allocator as "reserved but
unallocated". The scheduler's only error path kills the whole process tree, so
the engine died.

Returning that cached memory to the driver before a forward is enough to keep
the allocation inside the allocator's budget. It is also the only *safe* place
to act: the call is local to one rank, issues no collective, and therefore
cannot desynchronize tensor-parallel peers (catching the OOM mid-forward would:
the peers are already inside the same batch's collectives).

The CUDA calls are injected so this module can be unit tested without a GPU.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

# Throttle: the driver query is cheap but the decode loop runs every few ms.
CHECK_INTERVAL_S = float(os.environ.get("SGLANG_HEADROOM_GUARD_CHECK_S", "2.0"))
# Act once free memory on this rank drops below this.
FREE_THRESHOLD_MB = int(os.environ.get("SGLANG_HEADROOM_GUARD_FREE_MB", "768"))

_last_check_s = 0.0


def reset_for_tests() -> None:
    """Clear the throttle timestamp (tests only)."""
    global _last_check_s
    _last_check_s = 0.0


def guard_device_headroom(cuda, *, now=None) -> str:
    """Release cached device memory when headroom is low.

    `cuda` is a torch.cuda-like object exposing mem_get_info, empty_cache,
    memory_allocated and memory_reserved. Returns a short reason string so
    callers and tests can see what this call decided.
    """
    global _last_check_s
    now = time.monotonic() if now is None else now
    if now - _last_check_s < CHECK_INTERVAL_S:
        return "throttled"
    _last_check_s = now
    try:
        free, total = cuda.mem_get_info()
    except Exception:
        return "no-cuda"
    free_mb = free >> 20
    if free_mb >= FREE_THRESHOLD_MB:
        return "ok"
    try:
        cuda.empty_cache()
        free_after = cuda.mem_get_info()[0] >> 20
        allocated_mb = cuda.memory_allocated() >> 20
        reserved_mb = cuda.memory_reserved() >> 20
    except Exception:
        return "empty-cache-failed"
    logger.warning(
        "Low device headroom before forward: %d MiB free of %d MiB "
        "(threshold %d). empty_cache() returned %d MiB (now %d MiB free; "
        "allocated %d MiB, reserved %d MiB).",
        free_mb,
        total >> 20,
        FREE_THRESHOLD_MB,
        max(0, free_after - free_mb),
        free_after,
        allocated_mb,
        reserved_mb,
    )
    return "released"
