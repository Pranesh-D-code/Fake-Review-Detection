"""Share one page fetch between everything that needs the same page at the same moment.

The quick pass reads a product page for its price while the review collector reads the same page for its
review link, and the background pass later asks for the same review pages again. ``fetch_once`` makes the
second asker wait for (or reuse) the first answer instead of downloading it again. Entries live for two
minutes; a failed fetch is shared with whoever was waiting but is never kept.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

TTL_SECONDS = 120
MAX_ENTRIES = 200

_lock = threading.Lock()
_entries: dict[Any, "_Entry"] = {}


class _Entry:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None
        self.stored_at = time.time()


def fetch_once(key: Any, loader: Callable[[], Any], ttl: float = TTL_SECONDS) -> Any:
    with _lock:
        entry = _entries.get(key)
        if entry is not None and (not entry.done.is_set() or time.time() - entry.stored_at < ttl) and entry.error is None:
            owner = False
        else:
            entry = _Entry()
            if len(_entries) >= MAX_ENTRIES:
                oldest = min(_entries, key=lambda k: _entries[k].stored_at)
                _entries.pop(oldest, None)
            _entries[key] = entry
            owner = True
    if owner:
        try:
            entry.value = loader()
        except BaseException as exc:                    # shared with the waiters, then forgotten
            entry.error = exc
            with _lock:
                if _entries.get(key) is entry:
                    del _entries[key]
            raise
        finally:
            entry.stored_at = time.time()
            entry.done.set()
        return entry.value
    entry.done.wait(timeout=60)
    if entry.error is not None:
        raise entry.error
    return entry.value


def clear() -> None:
    with _lock:
        _entries.clear()
