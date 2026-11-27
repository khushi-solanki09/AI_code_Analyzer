# AI Code Analyzer

A Flask web app that reviews pasted source code: it flags bugs and security problems with rule-based checks, estimates time and space complexity, and predicts a quality label with a small machine-learning model.

Live demo: https://ai-code-analyzer-i6je.onrender.com/

Everything runs locally on the server. There are no calls to external APIs.

---

## What it does

| Module | How it works |
|---|---|
| Bug and security detection | 79 rules defined in `ml/rules/rules.yaml`. Code is parsed first: Python with `ast`, JavaScript/Java/C++ with tree-sitter, SQL with sqlparse. If the code does not parse, each rule falls back to a regular expression that runs on a copy with comments and string contents blanked. Every finding reports which engine produced it |
| Complexity estimate | Python is analysed with the `ast` module. JavaScript, Java and C++ use a brace-matching scanner. SQL gets code metrics only, because SQL cost depends on indexes and the query plan |
| Quality classifier | TF-IDF features with a Multinomial Naive Bayes model, trained on 151 hand-written samples. It returns "Good Code" or "Needs Improvement" with a confidence value |
| Ask about your analysis | A keyword-based helper that reads the latest analysis and answers from it. It is not an AI model |

Each finding shows a severity (High, Medium, Low), a short message and a suggested fix as text. The app does not rewrite your code automatically.

The browser also guesses the language from the code and warns if it does not match the selected tab. That check is a simple pattern test, not a parser.

### Rules per language

| Language | Rules | Of which parser-based | Engine |
|---|---|---|---|
| Python | 62 | 58 | `ast` |
| JavaScript | 16 | 15 | tree-sitter |
| Java | 15 | 14 | tree-sitter |
| C++ | 13 | 12 | tree-sitter |
| SQL | 9 | 3 | sqlparse |

A Python rule never runs on Java or SQL code. Several rules cover more than one language, so the numbers add up to more than 79. The remaining rules are text rules by nature (for example TODO comments) or have only a regex version for that language.

Parser-based rules understand the program: they resolve import aliases (`import pickle as p; p.load(x)` is caught), know which `break` belongs to which loop, and never look inside strings or comments. Python also gets checks a regex cannot do, such as unused imports and variables, too many parameters, duplicate dict keys and unreachable code.

### Rules are configuration

Rules are data in `ml/rules/rules.yaml` (id, severity, category, message, fix, and per-language implementations). `ml/rule_loader.py` validates the file at start-up, including compiling every regex and tree-sitter query, and refuses to start with a list of all problems. Adding a simple rule needs no Python code.

### Complexity cases handled

- Loops that halve or double their counter are O(log n).
- Loops over a constant range such as `range(10)` are O(1).
- Naive recursion with two calls is O(2^n). With `lru_cache` or a memo dict it is O(n).
- Divide-and-conquer recursion (binary search, merge sort) is O(log n) or O(n log n).
- Library calls with hidden cost are counted: `sorted()`, `.sort()`, and `sum()`/`max()`/`min()` inside a loop.

The result is a static estimate. It cannot know input sizes or what imported functions cost.

---

## Model evaluation

Run it yourself: `python -m ml.evaluate` (writes `ml/metrics.json`).

Method: repeated stratified 5-fold cross-validation, 10 repeats, on 151 samples (88 Good Code, 63 Needs Improvement).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Needs Improvement | 84.0% | 76.4% | 79.4% |
| Good Code | 84.6% | 89.2% | 86.6% |

Overall accuracy: 83.9% (standard deviation 5.6 across folds).

The classifier uses equal class priors. With the prior learned from the data, the model leaned towards "Good Code" (94% recall on that class but only 70% on Needs Improvement). Equal priors keep accuracy about the same (84.4% before, 83.9% after) and catch more bad code.

Limits of these numbers:
- 151 samples is small, so the scores are indicative, not proof of general quality.
- The samples were written by one person in one style, and the labels reflect one person's idea of quality.
- Real-world code will look different from these samples.

---

## API

`POST /analyze` with a JSON body:

```json
{
  "code": "x = eval(user_input)",
  "language": "Python",
  "run_bug": true,
  "run_security": true,
  "run_complex": true,
  "run_ml": true
}
```

- `code` is required, a string, at most 50,000 characters.
- `language` is one of Python, JavaScript, Java, C++, SQL (default Python).
- The four `run_*` options are optional booleans (default true).

Errors always come back as JSON: `{"error": "..."}` with status 400 (bad input), 413 (too large), 404, 405, 429 (rate limited) or 500.

Other endpoints: `GET /health` returns `{"status": "ok"}` and is never rate limited.

### Operations

| Setting (environment variable) | Meaning | Default |
|---|---|---|
| `ANALYZE_RATE_LIMIT` | Limit for `POST /analyze` per client IP | `30 per minute;500 per day` |
| `RATE_LIMIT_ENABLED` | `0` turns rate limiting off | on |
| `RATELIMIT_STORAGE_URI` | Where counters are kept | `memory://` |
| `TRUST_PROXY` | Number of reverse proxies in front of the app. Forwarded headers are believed only when this is set | `0` |
| `LOG_LEVEL` | Log level | `INFO` |

- Every response carries an `X-Request-ID` header (the caller's own id if it is safe, otherwise a generated one).
- Each request is logged as one JSON line (time, request id, method, path, status, duration). The submitted code is never logged, and unexpected errors log only the exception type.
- With the in-memory store each gunicorn worker keeps its own counters, so the effective limit is the setting times the number of workers. Use a shared store (Redis) through `RATELIMIT_STORAGE_URI` when you need an exact limit.

---

## Running locally

```bash
git clone https://github.com/khushi-solanki09/AI_code_Analyzer.git
cd AI_code_Analyzer
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Set `PORT` to use another port. Debug mode is off by default; turn it on for development only with `FLASK_DEBUG=1 python app.py`.

For production use a WSGI server, for example `gunicorn app:app`.

### Docker

```bash
docker compose up --build        # http://localhost:8000
```

The image runs gunicorn as an unprivileged user and has a health check on `/health`. The Dockerfile has not been built on the machine where this was written (no Docker daemon was available); the CI workflow builds it and smoke-tests the container on every push.

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
ruff check .
```

The suite has 255 tests. Every rule has at least one test where it fires and one where it stays quiet, plus tests for the rule file validation, the parsers' fallbacks, the `/analyze` endpoint, rate limiting and logging. GitHub Actions (`.github/workflows/ci.yml`) runs lint and tests on Python 3.11 to 3.13 and builds the Docker image.

## Project layout

```
app.py                     Flask app factory, validation, rate limiting, logging, JSON errors
ml/rules/rules.yaml        The 79 rules
ml/rule_loader.py          Loads and validates the rules
ml/bug_detector.py         Picks the engine for each rule and falls back to regex
ml/python_ast.py           Python ast checks
ml/treesitter_engine.py    tree-sitter queries for JavaScript, Java, C++
ml/sql_engine.py           sqlparse checks
ml/text_checks.py          Line-based fallbacks
ml/preprocess.py           Splits code into code / strings / comments
ml/complexity.py           Complexity estimator and code metrics
ml/classifier.py           Training data and the Naive Bayes pipeline
ml/evaluate.py             Cross-validation report -> ml/metrics.json
static/, templates/        Web interface
tests/                     pytest suite
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

---

## Known limitations

- Rules that run on tree-sitter or `ast` are structural, but the regex fallbacks (used when code does not parse) can still miss problems or raise false alarms.
- Only 3 of the 9 SQL rules are parser-based; SQL parsing in sqlparse is shallow.
- The non-Python complexity scanner is a heuristic.
- The classifier is trained on a small, hand-written dataset.
- Cross-line or cross-function data flow (for example user input reaching `eval` through several variables) is not tracked yet.
- Rate limiting is per process unless a shared store is configured.

## Planned improvements

- Taint tracking and OWASP/CWE mapping
- A larger, public training dataset
- Automatic fix suggestions as diffs
- Optional LLM-assisted explanations with evaluation
- GitHub pull-request integration, persistent storage and authentication

## Author

Khushi Singh Solanki
