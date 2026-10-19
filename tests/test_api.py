from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_classify_endpoint_returns_findings_and_summary():
    body = {"reviews": [
        {"text": "I love this product. It is very good and works very well. The quality is great and the price is also very good. I would recommend this product to anyone.", "rating": 5},
        {"text": "Delivery was late by 3 days and the box was torn, but the speaker itself works fine and the bass is decent for the price.", "rating": 4, "date": "Reviewed in India on 5 March 2024"},
        {"text": "ok", "rating": 3},
    ]}
    response = client.post("/api/reviews/classify", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["review_analysis"]["total_reviews"] == 3
    assert len(data["findings"]) == 3
    assert {"classification", "ml_fake_probability", "detector"} <= set(data["findings"][0])


def test_classify_endpoint_rejects_empty_input():
    assert client.post("/api/reviews/classify", json={"reviews": []}).status_code == 400
    assert client.post("/api/reviews/classify", json={"reviews": [{"text": "   "}]}).status_code == 400
