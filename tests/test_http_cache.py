import threading
import time

import pytest

from app import http_cache


def test_concurrent_askers_share_one_download():
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return "page"

    results = []
    threads = [threading.Thread(target=lambda: results.append(http_cache.fetch_once("k", slow))) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == ["page"] * 5 and len(calls) == 1


def test_a_failed_download_is_shared_with_waiters_but_not_kept():
    attempts = []

    def boom():
        attempts.append(1)
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        http_cache.fetch_once("bad", boom)
    assert http_cache.fetch_once("bad", lambda: "recovered") == "recovered" and len(attempts) == 1


def test_entries_expire():
    http_cache.fetch_once("ttl", lambda: 1, ttl=0)
    assert http_cache.fetch_once("ttl", lambda: 2, ttl=0) == 2
