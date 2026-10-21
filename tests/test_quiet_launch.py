"""Nothing the app starts may ever open a window."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_helper_the_app_launches_goes_through_the_hidden_window_helper():
    for path in sorted((ROOT / "app").glob("*.py")) + [ROOT / "scripts" / "auto_retrain.py"]:
        if path.name == "proc.py":
            continue
        source = path.read_text(encoding="utf-8")
        if re.search(r"subprocess\.(run|Popen)\(", source):
            assert "hidden_kwargs" in source, f"{path.name} starts a process without the hidden-window helper"
            assert "0x08000000" not in source and "STARTUPINFO" not in source, f"{path.name} builds its own window flags"


def test_every_browser_is_started_through_the_quiet_launcher():
    for name in ("scraper.js", "scraper_search.js", "scraper_reviews.js"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "launchQuiet" in source and "puppeteer.launch(" not in source, f"{name} launches Chrome itself"
    launcher = (ROOT / "browser_launch.js").read_text(encoding="utf-8")
    assert "headless: 'shell'" in launcher and "window-position" not in launcher
