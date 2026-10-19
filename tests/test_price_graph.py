"""Price comparison payload and the lowest-price rule, offline."""
from datetime import datetime, timedelta, timezone

from app.matcher import OUT_OF_STOCK_FLAG
from app.models import Candidate, Product
from app.price_graph import build_price_comparison, format_inr

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def source(platform="Croma", url="https://www.croma.com/m36/p/1", price=21999.0, in_stock=None, currency="INR"):
    product = Product(platform=platform, url=url, title="Samsung Galaxy M36 5G 8GB 128GB", price=price, currency=currency,
                      rating=4.3, review_count=120, image_url="https://img/src.jpg", category="electronics")
    product._p_data = {"in_stock": in_stock}
    return product


def cand(platform, price, status="verified", url=None, flags=(), title="Samsung Galaxy M36 5G 8GB 128GB", score=0.9, rating=4.1):
    return Candidate(
        platform=platform, title=title, url=url or f"https://{platform.lower().replace(' ', '')}.example/m36", price=price,
        rating=rating, title_similarity=0.8, status=status, reason="r", match_source="enhanced", match_score=score,
        match_reasons=["model code m36 matches"] if status == "verified" else [], match_flags=list(flags))


def no_history(url):
    return []


def build(product, candidates, loader=no_history, **kw):
    return build_price_comparison(product, candidates, history_loader=loader, now=NOW, **kw)


def test_lowest_is_the_cheapest_verified_in_stock_listing_across_all_stores():
    result = build(source(), [cand("Amazon", 20999.0), cand("Flipkart", 19499.0), cand("Vijay Sales", 20250.0)])
    assert result["lowest"]["store"] == "Flipkart" and result["lowest"]["price"] == 19499.0
    assert result["lowest_summary"].startswith("Lowest: Rs 19,499 at Flipkart")
    assert result["lowest_link"] == "https://flipkart.example/m36"
    assert [e["store"] for e in result["series"]] == ["Croma", "Amazon", "Flipkart", "Vijay Sales"]
    assert [e["is_lowest"] for e in result["series"]] == [False, False, True, False]


def test_out_of_stock_listing_is_shown_but_never_the_lowest():
    result = build(source(), [cand("Amazon", 15000.0, flags=[OUT_OF_STOCK_FLAG]), cand("Flipkart", 20500.0)])
    amazon = next(e for e in result["series"] if e["store"] == "Amazon")
    assert amazon["excluded_reason"] == "out of stock" and amazon["eligible"] is False and amazon["in_stock"] is False
    assert result["lowest"]["store"] == "Flipkart"
    assert any("Amazon" in note and "out of stock" in note for note in result["notes"])


def test_possible_matches_are_separate_and_never_the_lowest():
    result = build(source(), [cand("Amazon", 9000.0, status="candidate", flags=["storage differs"], score=0.55),
                              cand("Flipkart", 20500.0)])
    assert [e["store"] for e in result["series"]] == ["Croma", "Flipkart"]
    assert [e["store"] for e in result["possible"]] == ["Amazon"]
    assert result["possible"][0]["eligible"] is False and result["possible"][0]["kind"] == "possible"
    assert result["lowest"]["store"] == "Flipkart"
    assert any("never used as the lowest" in note for note in result["notes"])


def test_source_can_be_the_lowest():
    result = build(source(price=18000.0), [cand("Amazon", 20999.0)])
    assert result["lowest"]["store"] == "Croma" and result["lowest"]["kind"] == "source"
    assert result["lowest_summary"].endswith("(the listing you pasted)")


def test_out_of_stock_source_is_excluded_from_the_lowest():
    result = build(source(price=10000.0, in_stock=False), [cand("Amazon", 20999.0)])
    assert result["series"][0]["excluded_reason"] == "out of stock"
    assert result["lowest"]["store"] == "Amazon"


def test_refurbished_listing_cannot_be_the_lowest_for_a_new_source():
    result = build(source(), [cand("Amazon", 12000.0, title="Samsung Galaxy M36 5G 8GB 128GB (Renewed)"), cand("Flipkart", 20500.0)])
    amazon = next(e for e in result["series"] if e["store"] == "Amazon")
    assert amazon["excluded_reason"] == "refurbished / used listing"
    assert result["lowest"]["store"] == "Flipkart"


def test_no_eligible_listing_means_no_lowest_rather_than_a_guess():
    result = build(source(price=None), [cand("Amazon", 15000.0, flags=[OUT_OF_STOCK_FLAG]), cand("Flipkart", 9000.0, status="candidate", score=0.5)])
    assert result["lowest"] is None and result["lowest_summary"] is None and result["lowest_link"] is None


def test_one_listing_per_store_the_cheapest_buyable_one():
    result = build(source(), [cand("Amazon", 15000.0, flags=[OUT_OF_STOCK_FLAG], url="https://amazon.example/a"),
                              cand("Amazon", 21000.0, url="https://amazon.example/b"),
                              cand("Amazon", 22000.0, url="https://amazon.example/c")])
    amazons = [e for e in result["series"] if e["store"] == "Amazon"]
    assert len(amazons) == 1 and amazons[0]["url"] == "https://amazon.example/b"
    assert any("more verified listing" in note for note in result["notes"])


def test_the_source_store_is_never_repeated_as_a_match():
    result = build(source(), [cand("Croma", 15000.0, url="https://www.croma.com/other/p/9"), cand("Amazon", 21000.0)])
    assert [e["store"] for e in result["series"]] == ["Croma", "Amazon"]


def test_foreign_currency_source_is_left_out_of_the_comparison():
    result = build(source(price=250.0, currency="USD"), [cand("Amazon", 20999.0)])
    assert result["series"][0]["eligible"] is False and "USD" in result["series"][0]["excluded_reason"]
    assert result["lowest"]["store"] == "Amazon"


def test_points_are_only_real_observations_plus_the_live_price():
    history = {
        "https://www.croma.com/m36/p/1": [(NOW - timedelta(days=20), 23999.0), (NOW - timedelta(days=5), 22499.0)],
        "https://flipkart.example/m36": [(NOW - timedelta(days=2), 20999.0)],
    }
    result = build(source(), [cand("Flipkart", 19499.0), cand("Amazon", 20999.0)], loader=lambda url: history.get(url, []))
    by_store = {e["store"]: e for e in result["series"]}
    assert [p["price"] for p in by_store["Croma"]["points"]] == [23999.0, 22499.0, 21999.0]
    assert by_store["Croma"]["points"][-1]["observed_at"] == NOW.isoformat()
    assert [p["price"] for p in by_store["Flipkart"]["points"]] == [20999.0, 19499.0]
    assert len(by_store["Amazon"]["points"]) == 1            # never seen before: one point, no back-filled history
    assert result["observed_history_points"] == 3 + 2 + 1


def test_a_price_just_recorded_is_not_plotted_twice():
    recorded = {"https://www.croma.com/m36/p/1": [(NOW - timedelta(minutes=2), 21999.0)]}
    result = build(source(), [], loader=lambda url: recorded.get(url, []))
    assert len(result["series"][0]["points"]) == 1


def test_a_failing_history_lookup_still_returns_the_live_point():
    def broken(url):
        raise RuntimeError("db locked")

    result = build(source(), [cand("Amazon", 20999.0)], loader=broken)
    assert all(len(e["points"]) == 1 for e in result["series"])


def test_legacy_history_synthetic_flag_follows_the_stored_row_count():
    one_row = lambda url: [(NOW - timedelta(minutes=1), 21999.0)]
    two_rows = lambda url: [(NOW - timedelta(days=3), 22999.0), (NOW - timedelta(minutes=1), 21999.0)]
    assert build(source(), [], loader=one_row, legacy_summary_present=True)["legacy_history_synthetic"] is True
    assert build(source(), [], loader=two_rows, legacy_summary_present=True)["legacy_history_synthetic"] is False
    assert build(source(), [], loader=one_row, legacy_summary_present=False)["legacy_history_synthetic"] is False


def test_stores_searched_reports_what_each_store_returned():
    attempts = [{"platform": "Croma", "status": "results collected", "result_count": 1, "raw_results": 4},
                {"platform": "Reliance Digital", "status": "blocked", "result_count": 0, "raw_results": 0}]
    result = build(source(platform="Amazon", url="https://www.amazon.in/dp/B0X"), [], attempts=attempts)
    assert [(s["store"], s["status"], s["listings"]) for s in result["stores_searched"]] == [
        ("Croma", "results collected", 1), ("Reliance Digital", "blocked", 0)]


def test_amazon_source_keeps_its_legacy_label():
    result = build(source(platform="Amazon", url="https://www.amazon.in/dp/B0X"), [cand("Flipkart", 20000.0)])
    assert result["source_store"] == "Amazon"


def test_inr_formatting_uses_indian_grouping():
    assert format_inr(24999) == "Rs 24,999" and format_inr(124999) == "Rs 1,24,999"
    assert format_inr(999) == "Rs 999" and format_inr(1999.5) == "Rs 1,999.50" and format_inr(None) == "price not read"


def test_matcher_condition_flag_keeps_a_second_hand_listing_out_of_the_lowest():
    result = build(source(), [cand("Amazon", 12000.0, flags=["refurbished/used listing"]), cand("Flipkart", 20500.0)])
    assert next(e for e in result["series"] if e["store"] == "Amazon")["excluded_reason"] == "refurbished / used listing"
    assert result["lowest"]["store"] == "Flipkart"


def test_a_store_reported_by_both_searches_is_listed_once():
    attempts = [{"platform": "Amazon", "status": "results collected", "result_count": 6},
                {"platform": "Amazon", "status": "unavailable: HTTP 503", "result_count": 0},
                {"platform": "Flipkart", "status": "results collected", "result_count": 3}]
    result = build(source(), [], attempts=attempts)
    assert [(s["store"], s["listings"]) for s in result["stores_searched"]] == [("Amazon", 6), ("Flipkart", 3)]
