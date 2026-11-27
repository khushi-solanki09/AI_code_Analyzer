"""Flask web server: connects the analysis modules to the web UI.

Operational behaviour (all configurable through environment variables):

    ANALYZE_RATE_LIMIT      limit for POST /analyze per client IP
                            (default "30 per minute;500 per day")
    RATE_LIMIT_ENABLED      "0" switches rate limiting off (default on)
    RATELIMIT_STORAGE_URI   where counters live (default memory://, which is
                            per process; use redis://... with several workers)
    TRUST_PROXY             number of reverse proxies in front of the app. Only
                            then are X-Forwarded-* headers believed, so the
                            limiter sees the real client IP and not the proxy
    LOG_LEVEL               default INFO

Every request gets an id (X-Request-ID, taken from the caller or generated),
is logged as one JSON line, and the submitted code is never logged.
"""

import json
import logging
import os
import re
import sys
import time
import uuid

from flask import Flask, g, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from ml.bug_detector import detect_bugs, get_bug_summary, rules_for
from ml.classifier import classify_code
from ml.complexity import analyze_complexity
from ml.preprocess import DISPLAY_NAMES, SUPPORTED_LANGUAGES, normalize_language

MAX_CODE_CHARS = 50_000          # about 1,500 lines of code
MAX_BODY_BYTES = 256 * 1024      # hard limit on the whole request

DEFAULT_ANALYZE_LIMIT = "30 per minute;500 per day"
_REQUEST_ID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

log = logging.getLogger("analyzer")


class JsonFormatter(logging.Formatter):
    """One JSON object per log line. Extra fields passed via `extra=` are kept."""

    RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record):
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.RESERVED:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is at the moment of the call."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def _setup_logging():
    if log.handlers:
        return
    handler = _StdoutHandler()
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    log.propagate = False



class BadRequest(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def _flag(data, key):
    """Optional boolean option, True when missing."""
    value = data.get(key, True)
    if not isinstance(value, bool):
        raise BadRequest("'%s' must be true or false." % key)
    return value


def _parse_request():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object, e.g. {\"code\": \"...\", \"language\": \"Python\"}.")

    code = data.get("code")
    if not isinstance(code, str):
        raise BadRequest("'code' must be a string.")
    if not code.strip():
        raise BadRequest("Code is empty!")
    if len(code) > MAX_CODE_CHARS:
        raise BadRequest("Code is too long (%d characters). The limit is %d." % (len(code), MAX_CODE_CHARS), 413)

    language = normalize_language(data.get("language", "Python"))
    if language is None:
        supported = ", ".join(DISPLAY_NAMES[l] for l in SUPPORTED_LANGUAGES)
        raise BadRequest("Unsupported language. Supported: %s." % supported)

    return {
        "code": code,
        "language": language,
        "run_bug": _flag(data, "run_bug"),
        "run_complex": _flag(data, "run_complex"),
        "run_security": _flag(data, "run_security"),
        "run_ml": _flag(data, "run_ml"),
    }


def create_app(analyze_limit=None, rate_limit_enabled=None, storage_uri=None):
    """Build the Flask app. Arguments override the environment (used by tests)."""
    _setup_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES

    proxies = int(os.environ.get("TRUST_PROXY", "0") or 0)
    if proxies > 0:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxies, x_proto=proxies, x_host=proxies)

    if rate_limit_enabled is None:
        rate_limit_enabled = os.environ.get("RATE_LIMIT_ENABLED", "1") != "0"
    analyze_limit = analyze_limit or os.environ.get("ANALYZE_RATE_LIMIT", DEFAULT_ANALYZE_LIMIT)
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri=storage_uri or os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
        enabled=rate_limit_enabled,
        headers_enabled=True,
    )
    app.limiter = limiter  # Flask-Limiter keeps only a weak reference to itself

    # ===== request id + one JSON log line per request =====
    @app.before_request
    def start_request():
        incoming = request.headers.get("X-Request-ID", "")
        g.request_id = incoming if _REQUEST_ID_OK.match(incoming) else uuid.uuid4().hex[:16]
        g.started = time.perf_counter()

    @app.after_request
    def finish_request(response):
        response.headers["X-Request-ID"] = g.get("request_id", "-")
        if request.endpoint != "static":
            log.info("request", extra={
                "request_id": g.get("request_id", "-"),
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - g.get("started", time.perf_counter())) * 1000, 1),
                "ip": request.remote_addr,
                "bytes_in": request.content_length or 0,
            })
        return response

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/health")
    @limiter.exempt
    def health():
        return jsonify({"status": "ok"})

    @app.route("/analyze", methods=["POST"])
    @limiter.limit(analyze_limit)
    def analyze():
        """Analyze code with the modules the caller switched on."""
        opts = _parse_request()
        code, language = opts["code"], opts["language"]

        # 1. Bug detection (also used for the security-only view)
        if opts["run_bug"] or opts["run_security"]:
            bugs = detect_bugs(code, language)
            if not opts["run_bug"]:
                bugs = [b for b in bugs if b["category"] == "Security"]
            bug_summary = get_bug_summary(bugs)
        else:
            bugs = []
            bug_summary = {"total": 0, "high": 0, "medium": 0, "low": 0}

        # 2. Complexity analysis
        if opts["run_complex"]:
            complexity = analyze_complexity(code, language)
        else:
            complexity = {
                "total_lines": 0, "code_lines": 0, "comment_lines": 0,
                "function_count": 0, "class_count": 0, "loop_count": 0,
                "condition_count": 0, "max_nesting": 0, "complexity_score": 0,
                "quality_score": 0, "time_complexity": "N/A",
                "space_complexity": "N/A", "recursive_calls": 0,
                "data_structures_used": 0,
            }

        # 3. ML classification
        if opts["run_ml"]:
            classification = classify_code(code)
        else:
            classification = {"quality": "N/A", "confidence": 0, "score": 0}

        return jsonify({
            "bugs": bugs,
            "bug_summary": bug_summary,
            "complexity": complexity,
            "classification": classification,
            "language": DISPLAY_NAMES[language],
            "rules_checked": len(rules_for(language)),
            "modules_run": {
                "bug": opts["run_bug"],
                "security": opts["run_security"],
                "complexity": opts["run_complex"],
                "ml": opts["run_ml"],
            },
        })

    # ===== ERRORS: always answer in JSON =====
    @app.errorhandler(BadRequest)
    def handle_bad_request(err):
        return jsonify({"error": err.message}), err.status

    @app.errorhandler(413)
    def handle_too_large(_err):
        return jsonify({"error": "Request is too large. The limit is %d KB." % (MAX_BODY_BYTES // 1024)}), 413

    @app.errorhandler(429)
    def handle_rate_limited(err):
        response = jsonify({"error": "Too many requests. Please wait a moment and try again."})
        response.status_code = 429
        retry = getattr(err, "retry_after", None)
        if retry:
            response.headers["Retry-After"] = str(int(retry))
        return response

    @app.errorhandler(HTTPException)
    def handle_http_error(err):
        return jsonify({"error": err.description or err.name}), err.code

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        # Log the exception type and request id, never the submitted code.
        log.error("unhandled_error", extra={
            "request_id": g.get("request_id", "-"),
            "error_type": type(err).__name__,
        }, exc_info=True)
        return jsonify({"error": "Something went wrong while analyzing the code."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    # Debug mode exposes an interactive debugger, so it is opt-in:
    #   FLASK_DEBUG=1 python app.py
    debug = os.environ.get("FLASK_DEBUG") == "1"
    port = int(os.environ.get("PORT", "5000"))
    print("Open in browser: http://localhost:%d" % port)
    app.run(debug=debug, port=port)
