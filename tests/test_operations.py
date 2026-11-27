"""Rate limiting, request ids, structured logging and error handling."""

import json
import logging

import pytest

from app import JsonFormatter, create_app

GOOD = {"code": "x = 1\n", "language": "Python"}


@pytest.fixture()
def limited():
    app = create_app(analyze_limit="3 per minute", rate_limit_enabled=True)
    return app.test_client()


def test_fourth_request_is_rate_limited_with_json(limited):
    for _ in range(3):
        assert limited.post("/analyze", json=GOOD).status_code == 200
    r = limited.post("/analyze", json=GOOD)
    assert r.status_code == 429
    assert "error" in r.get_json()
    assert int(r.headers["Retry-After"]) >= 1


def test_limit_is_per_client_ip(limited):
    for _ in range(3):
        limited.post("/analyze", json=GOOD)
    assert limited.post("/analyze", json=GOOD).status_code == 429
    other = limited.post("/analyze", json=GOOD, environ_overrides={"REMOTE_ADDR": "10.9.9.9"})
    assert other.status_code == 200


def test_home_and_health_are_not_limited(limited):
    for _ in range(10):
        assert limited.get("/health").status_code == 200
        assert limited.get("/").status_code == 200
    assert limited.get("/health").get_json() == {"status": "ok"}


def test_rate_limit_can_be_disabled():
    client = create_app(analyze_limit="1 per minute", rate_limit_enabled=False).test_client()
    assert all(client.post("/analyze", json=GOOD).status_code == 200 for _ in range(5))


def test_forwarded_header_ignored_without_trust_proxy(limited):
    for _ in range(3):
        limited.post("/analyze", json=GOOD)
    r = limited.post("/analyze", json=GOOD, headers={"X-Forwarded-For": "8.8.8.8"})
    assert r.status_code == 429  # spoofing a new IP does not reset the limit


def test_forwarded_header_used_behind_trusted_proxy(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    client = create_app(analyze_limit="1 per minute", rate_limit_enabled=True).test_client()
    assert client.post("/analyze", json=GOOD, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/analyze", json=GOOD, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    assert client.post("/analyze", json=GOOD, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


def test_request_id_generated_and_echoed(limited):
    r = limited.post("/analyze", json=GOOD)
    assert len(r.headers["X-Request-ID"]) >= 8
    r2 = limited.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r2.headers["X-Request-ID"] == "abc-123"


def test_unsafe_request_id_is_replaced(limited):
    r = limited.get("/health", headers={"X-Request-ID": "bad id\twith spaces"})
    assert r.headers["X-Request-ID"] != "bad id\twith spaces"


def test_request_is_logged_as_json_without_the_code(limited, capsys):
    secret = "password = 'hunter2-should-never-be-logged'"
    limited.post("/analyze", json={"code": secret, "language": "Python"},
                 headers={"X-Request-ID": "req-1"})
    out = capsys.readouterr().out
    entries = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    entry = next(e for e in entries if e["event"] == "request")
    assert entry["request_id"] == "req-1"
    assert entry["method"] == "POST" and entry["path"] == "/analyze" and entry["status"] == 200
    assert entry["duration_ms"] >= 0
    assert "hunter2" not in out


def test_unhandled_error_is_json_logged_and_hides_details(monkeypatch, capsys):
    import app as app_module

    def boom(*_a, **_k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app_module, "detect_bugs", boom)
    client = create_app(rate_limit_enabled=False).test_client()
    r = client.post("/analyze", json=GOOD)
    assert r.status_code == 500
    assert r.get_json() == {"error": "Something went wrong while analyzing the code."}
    assert "secret internal detail" not in r.get_data(as_text=True)
    assert r.headers["X-Request-ID"]
    out = capsys.readouterr().out
    entry = next(json.loads(l) for l in out.splitlines() if '"unhandled_error"' in l)
    assert entry["error_type"] == "RuntimeError"


def test_errors_stay_json_for_404_and_405(limited):
    assert limited.get("/nope").get_json()["error"]
    r = limited.get("/analyze")
    assert r.status_code == 405 and "error" in r.get_json()


def test_json_formatter_includes_extra_fields():
    rec = logging.LogRecord("t", logging.INFO, "f", 1, "hello", None, None)
    rec.request_id = "r9"
    data = json.loads(JsonFormatter().format(rec))
    assert data["event"] == "hello" and data["request_id"] == "r9" and data["level"] == "INFO"
