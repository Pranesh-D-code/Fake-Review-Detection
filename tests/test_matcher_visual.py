"""Text and photo evidence together (offline: photo verdicts are passed in, images are synthetic)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import visual_match
from app.matcher import score_candidate
from app.product_identity import parse_identity

SOURCE = parse_identity("Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB Storage) | 6.7 inch Super AMOLED", "Samsung")


def verdict(title, visual, similarity=0.9, url=None, price=21999, source=SOURCE):
    return score_candidate(source, title, url, 22999, price, similarity, visual_verdict=visual)


def test_a_matching_photo_confirms_a_title_that_is_only_partly_there():
    # no model code in the title, name words present: the text alone leaves it a candidate
    text_only = verdict("Samsung Galaxy 5G Smartphone 8GB 128GB", None, similarity=None)
    both = verdict("Samsung Galaxy 5G Smartphone 8GB 128GB", "same")
    assert text_only.status != "verified"
    assert both.status in {"candidate", "verified"} and both.score > text_only.score


def test_a_different_photo_stops_a_text_match_from_being_verified():
    title = "Samsung Galaxy M36 5G (8GB RAM, 128GB, Velvet Black)"
    assert verdict(title, "same").status == "verified"
    assert verdict(title, None, similarity=None).status == "verified"
    doubtful = verdict(title, "different", similarity=0.1)
    assert doubtful.status == "candidate" and any("photo looks different" in f for f in doubtful.flags)


def test_the_photo_can_never_override_a_clear_text_contradiction():
    assert verdict("Back Case for Samsung Galaxy M36 5G", "same").status == "rejected"            # accessory
    assert verdict("Samsung Galaxy A36 5G (8GB RAM, 128GB)", "same").status == "rejected"         # another model
    wrong_storage = verdict("Samsung Galaxy M36 5G (8GB RAM, 256GB Storage)", "same")
    assert wrong_storage.status == "candidate" and any("specification" in f for f in wrong_storage.flags)


def test_without_a_verdict_the_old_numeric_behaviour_is_unchanged():
    base = verdict("Samsung Galaxy M36 5G (8GB RAM, 128GB)", None, similarity=None).score
    assert verdict("Samsung Galaxy M36 5G (8GB RAM, 128GB)", None, similarity=0.85).score == pytest.approx(min(1.0, base + 0.10), abs=0.002)


# ---- the image helpers --------------------------------------------------------------------------------

def product_photo(colour=(200, 30, 30), size=(300, 400), box=(90, 120, 210, 300), background=(255, 255, 255)):
    image = Image.new("RGB", size, background)
    ImageDraw.Draw(image).rectangle(box, fill=colour)
    return image


def test_background_is_trimmed_and_the_product_is_centred():
    trimmed = visual_match.trim_to_product(product_photo())
    array = np.asarray(trimmed)
    assert trimmed.size == (224, 224)
    red = (array[:, :, 0] > 150) & (array[:, :, 1] < 80)
    assert red.mean() > 0.4                          # the product now fills about half the frame (it was 14% of the photo)


def test_same_object_at_different_scale_matches_and_another_colour_does_not(monkeypatch):
    monkeypatch.setattr(visual_match, "_load_model", lambda: None)                  # hash fallback: deterministic, no model needed
    big = product_photo(box=(60, 80, 240, 340))
    small = product_photo(box=(110, 160, 190, 280))
    other = product_photo(colour=(20, 20, 200), box=(60, 200, 250, 260))
    same = visual_match.compare_images(big, small, "a", "b")
    different = visual_match.compare_images(big, other, "a", "c")
    assert same.method == "phash" and same.score > different.score
    assert same.colour > 0.9 and different.colour < 0.2                             # colour tells the red item from the blue one


def test_unreachable_images_give_no_score():
    assert visual_match.compare("not a url", None) is None
    assert visual_match.compare(None, "https://x.example/a.jpg") is None


def test_calibration_defaults_work_without_a_file(monkeypatch):
    monkeypatch.setattr(visual_match, "_calibration", {})
    assert 0.0 < visual_match.probability_from_features(0.5, 0.5, 0.5) < 1.0
    assert visual_match.verdict_for(0.97, "dinov2") == "same" and visual_match.verdict_for(0.05, "dinov2") == "different"
    assert visual_match.verdict_for(0.5, "dinov2") == "unclear"
