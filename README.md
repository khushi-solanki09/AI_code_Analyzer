# AI Code Analyzer

A Flask web app that reviews pasted source code: it flags bugs and security problems with rule-based checks, estimates time and space complexity, and predicts a quality label with a small machine-learning model.

Live demo: https://ai-code-analyzer-i6je.onrender.com/

Everything runs locally on the server. There are no calls to external APIs.

---

## What it does

| Module | How it works |
|---|---|
| Bug and security detection | 69 regular-expression rules, each tied to the languages it applies to. Comments and string contents are removed before matching, so words inside them never trigger a rule |
| Complexity estimate | Python is analysed with the `ast` module. JavaScript, Java and C++ use a brace-matching scanner. SQL gets code metrics only, because SQL cost depends on indexes and the query plan |
| Quality classifier | TF-IDF features with a Multinomial Naive Bayes model, trained on 151 hand-written samples. It returns "Good Code" or "Needs Improvement" with a confidence value |
| Ask about your analysis | A keyword-based helper that reads the latest analysis and answers from it. It is not an AI model |

Each finding shows a severity (High, Medium, Low), a short message and a suggested fix as text. The app does not rewrite your code automatically.

The browser also guesses the language from the code and warns if it does not match the selected tab. That check is a simple pattern test, not a parser.

### Rules per language

| Language | Rules that apply |
|---|---|
| Python | 54 |
| JavaScript | 16 |
| Java | 14 |
| C++ | 13 |
| SQL | 8 |

A Python rule never runs on Java or SQL code. Several rules cover more than one language, so the numbers add up to more than 69.

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

Errors always come back as JSON: `{"error": "..."}` with status 400 (bad input), 413 (too large), 404, 405 or 500.

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

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project layout

```
app.py                 Flask server, input validation, JSON errors
ml/preprocess.py       Splits code into code / strings / comments
ml/bug_detector.py     The rule table and the matcher
ml/complexity.py       Complexity estimator and code metrics
ml/classifier.py       Training data and the Naive Bayes pipeline
ml/evaluate.py         Cross-validation report -> ml/metrics.json
static/, templates/    Web interface
tests/                 pytest suite
```

---

## Known limitations

- Bug detection is pattern matching, not full parsing. It can miss problems and can still raise false alarms on unusual code.
- The non-Python complexity scanner is a heuristic.
- The classifier is trained on a small, hand-written dataset.
- Cross-line or cross-function data flow (for example user input reaching `eval` through several variables) is not tracked.

## Planned improvements

- AST and tree-sitter based rules for every language
- Rules loaded from configuration files
- A larger, public training dataset
- CI, Docker and a larger test suite

## Author

Khushi Singh Solanki
