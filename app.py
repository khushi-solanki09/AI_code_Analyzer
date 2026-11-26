"""Flask web server: connects the analysis modules to the web UI."""

import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from ml.bug_detector import detect_bugs, get_bug_summary, rules_for
from ml.classifier import classify_code
from ml.complexity import analyze_complexity
from ml.preprocess import DISPLAY_NAMES, SUPPORTED_LANGUAGES, normalize_language

MAX_CODE_CHARS = 50_000          # about 1,500 lines of code
MAX_BODY_BYTES = 256 * 1024      # hard limit on the whole request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES


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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
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


@app.errorhandler(HTTPException)
def handle_http_error(err):
    return jsonify({"error": err.description or err.name}), err.code


@app.errorhandler(Exception)
def handle_unexpected(err):
    app.logger.exception("Unhandled error: %s", err)
    return jsonify({"error": "Something went wrong while analyzing the code."}), 500


if __name__ == "__main__":
    # Debug mode exposes an interactive debugger, so it is opt-in:
    #   FLASK_DEBUG=1 python app.py
    debug = os.environ.get("FLASK_DEBUG") == "1"
    port = int(os.environ.get("PORT", "5000"))
    print("Open in browser: http://localhost:%d" % port)
    app.run(debug=debug, port=port)
