"""Stores are shown as each one answers, not after the slowest one (offline: the browser helper is faked)."""
import json

from app import multistore

TITLE = "Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB Storage)"


class FakeProc:
    def __init__(self, lines, finish=True):
        self.stdout = iter(lines)
        self._finish = finish
        self.killed = False

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0 if self._finish else None

    def kill(self):
        self.killed = True


def item(platform, title, price, url):
    return {"platform": platform, "title": title, "url": url, "price": price, "image_url": None}


STORE_A = {"store_done": {"platform": "Vijay Sales", "status": "ok", "result_count": 2, "query": "q"},
           "items": [item("Vijay Sales", "Samsung Galaxy M36 5G 8GB 128GB", 21999, "https://www.vijaysales.com/p/1/2/m36"),
                     item("Vijay Sales", "Back Case for Samsung Galaxy M36 5G", 399, "https://www.vijaysales.com/p/1/3/case")]}
STORE_B = {"store_done": {"platform": "Croma", "status": "ok", "result_count": 1, "query": "q"},
           "items": [item("Croma", "SAMSUNG Galaxy M36 5G (8GB RAM, 128GB, Black)", 22499, "https://www.croma.com/m36/p/9")]}
FINAL = {"stores": [STORE_A["store_done"], STORE_B["store_done"]], "matches": STORE_A["items"] + STORE_B["items"]}


def run(monkeypatch, lines, progress):
    monkeypatch.setattr(multistore.subprocess, "Popen", lambda *a, **k: FakeProc([json.dumps(x) + "\n" for x in lines]))
    monkeypatch.setattr(multistore, "select_stores", lambda *a, **k: ["Vijay Sales", "Croma"])
    monkeypatch.setattr(multistore, "_run_node", lambda mode, payload, timeout: {"details": []})         # the page-visit step
    monkeypatch.setattr(multistore, "compute_image_similarity", lambda a, b: None)
    monkeypatch.setattr(multistore, "_page_image", lambda url: None)
    return multistore.enhanced_competitor_matches(TITLE, "Samsung", 21999.0, None, "Amazon", "https://www.amazon.in/dp/B0X", progress=progress)


def test_each_store_is_reported_the_moment_it_answers(monkeypatch):
    heard = []
    matches, attempts, _ = run(monkeypatch, [STORE_A, STORE_B, FINAL], lambda m, a, i: heard.append(([x["platform"] for x in m], [x["platform"] for x in a])))
    assert heard[0] == (["Vijay Sales"], ["Vijay Sales"])                    # the first store is on screen before the second answers
    assert heard[1] == (["Vijay Sales", "Croma"], ["Vijay Sales", "Croma"])  # then the list grows
    assert {m["platform"] for m in matches} == {"Vijay Sales", "Croma"}
    assert all("Case" not in m["title"] for m in matches)                    # the accessory was filtered out of the early view and the final one
    assert {a["platform"] for a in attempts} == {"Vijay Sales", "Croma"}


def test_a_helper_that_is_cut_off_still_keeps_the_stores_that_answered(monkeypatch):
    heard = []
    matches, attempts, _ = run(monkeypatch, [STORE_A], lambda m, a, i: heard.append(len(m)))       # no final line: the helper was killed
    assert heard == [1] and [m["platform"] for m in matches] == ["Vijay Sales"]


def test_without_a_progress_callback_the_old_blocking_path_is_used(monkeypatch):
    monkeypatch.setattr(multistore.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not stream")))
    monkeypatch.setattr(multistore, "select_stores", lambda *a, **k: ["Croma"])
    monkeypatch.setattr(multistore, "_run_node", lambda mode, payload, timeout: FINAL if mode == "search" else {"details": []})
    monkeypatch.setattr(multistore, "compute_image_similarity", lambda a, b: None)
    monkeypatch.setattr(multistore, "_page_image", lambda url: None)
    matches, _, _ = multistore.enhanced_competitor_matches(TITLE, "Samsung", 21999.0, None, "Amazon", "https://www.amazon.in/dp/B0X")
    assert matches
