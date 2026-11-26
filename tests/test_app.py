import json

import pytest

from app import MAX_CODE_CHARS, app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


def post(client, payload, raw=False):
    if raw:
        return client.post("/analyze", data=payload, content_type="application/json")
    return client.post("/analyze", json=payload)


def test_home_page(client):
    assert client.get("/").status_code == 200


def test_analyze_ok(client):
    r = post(client, {"code": "x = eval(s)\n", "language": "Python"})
    body = r.get_json()
    assert r.status_code == 200
    assert body["language"] == "Python" and body["bug_summary"]["high"] == 1
    assert {"bugs", "complexity", "classification", "rules_checked"} <= set(body)


def test_language_is_used_by_the_backend(client):
    py = post(client, {"code": "x = eval(s)\n", "language": "Python"}).get_json()
    java = post(client, {"code": "x = eval(s)\n", "language": "Java"}).get_json()
    assert py["bug_summary"]["total"] > java["bug_summary"]["total"]


def test_default_language_is_python(client):
    assert post(client, {"code": "x = eval(s)\n"}).get_json()["language"] == "Python"


def test_security_only_filters_other_categories(client):
    r = post(client, {"code": 'password = "x"\nprint(1)\n', "run_bug": False, "run_security": True}).get_json()
    assert r["bugs"] and all(b["category"] == "Security" for b in r["bugs"])


def test_modules_can_be_switched_off(client):
    r = post(client, {"code": "x = 1\n", "run_bug": False, "run_security": False,
                      "run_complex": False, "run_ml": False}).get_json()
    assert r["bugs"] == [] and r["complexity"]["time_complexity"] == "N/A" and r["classification"]["quality"] == "N/A"


@pytest.mark.parametrize("payload,raw", [
    ("not json at all", True),
    ("[1, 2, 3]", True),
    ("null", True),
    ('{"code": 123}', True),
    ('{"code": ""}', True),
    ('{"code": "   "}', True),
    ('{"language": "Python"}', True),
    ('{"code": "x", "language": "Rust"}', True),
    ('{"code": "x", "language": 5}', True),
    ('{"code": "x", "run_bug": "yes"}', True),
])
def test_bad_requests_get_a_json_400(client, payload, raw):
    r = post(client, payload, raw=raw)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_code_too_long(client):
    r = post(client, {"code": "x = 1\n" * (MAX_CODE_CHARS // 4)})
    assert r.status_code == 413 and "error" in r.get_json()


def test_body_too_large(client):
    r = post(client, json.dumps({"code": "a" * (300 * 1024)}), raw=True)
    assert r.status_code == 413 and "error" in r.get_json()


def test_wrong_method_and_missing_page_are_json(client):
    assert client.get("/analyze").status_code == 405 and "error" in client.get("/analyze").get_json()
    assert client.get("/nope").status_code == 404 and "error" in client.get("/nope").get_json()


def test_debug_mode_is_off_by_default():
    assert app.debug is False
