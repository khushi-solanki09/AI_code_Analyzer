"""Rule-based bug detector.

How it works
------------
1. The source is split into three views (see preprocess.scan): code without
   comments, code with string contents blanked out, and comments only.
   Rules run on the view that suits them, so a word inside a comment or a
   string no longer triggers a rule by accident.
2. Every rule declares which languages it applies to. A Python rule never
   runs on Java or SQL code.
3. Rules are regular expressions, not bare substrings, so they can look at
   word boundaries and context (for example `password = os.environ[...]`
   is fine, `password = "admin123"` is not).

This is still pattern matching, not a parser: the Python AST engine is the
next planned upgrade (see the project roadmap).
"""

import re

from ml.preprocess import normalize_language, scan

PY, JS, JAVA, CPP, SQL = "python", "javascript", "java", "cpp", "sql"
CODE_LANGS = (PY, JS, JAVA, CPP)
ALL_LANGS = CODE_LANGS + (SQL,)


# ──────────────────────────────────────────────────────────────────────
# Helpers used by rules that need a little more than one regex
# ──────────────────────────────────────────────────────────────────────

def _indent(line):
    return len(line) - len(line.lstrip())


def _in_loop(lines, i):
    """True if line i sits (at any depth) inside a Python for/while loop."""
    ind = _indent(lines[i])
    for j in range(i - 1, -1, -1):
        s = lines[j]
        if not s.strip():
            continue
        if _indent(s) < ind:
            if re.match(r"\s*(?:async\s+)?(?:for|while)\b", s):
                return True
            ind = _indent(s)
            if ind == 0:
                break
    return False


_EXIT_RX = re.compile(r"\b(?:break|return|raise)\b|\bsys\.exit\b|\bexit\s*\(|\bos\._exit\b")


def _loop_has_exit(lines, i):
    """True if the Python loop that starts on line i contains break/return/raise."""
    base = _indent(lines[i])
    for j in range(i + 1, len(lines)):
        s = lines[j]
        if not s.strip():
            continue
        if _indent(s) <= base:
            break
        if _EXIT_RX.search(s):
            return True
    return False


_DEF_RX = re.compile(r"^(\s*)def\s+(\w+)\s*\(([^)]*)\)")


def _check_missing_negative_guard(lines):
    """Recursive function whose only base case is `n == 0`: a negative input
    recurses until RecursionError."""
    hits = []
    for i, line in enumerate(lines):
        m = _DEF_RX.match(line)
        if not m:
            continue
        base, name = len(m.group(1)), m.group(2)
        params = [p.split("=")[0].split(":")[0].strip() for p in m.group(3).split(",")]
        body = []
        j = i + 1
        while j < len(lines):
            s = lines[j]
            if s.strip() and _indent(s) <= base:
                break
            body.append((j, s))
            j += 1
        text = "\n".join(s for _, s in body)
        if not re.search(r"(?<![\w.])%s\s*\(" % re.escape(name), text):
            continue  # not recursive
        for p in params:
            if not p or p in ("self", "cls"):
                continue
            eq = [j for j, s in body if re.search(r"\b%s\s*==\s*0\b" % re.escape(p), s)]
            if not eq:
                continue
            guarded = (re.search(r"\b%s\s*(?:<|<=|>|>=)\s*-?\d" % re.escape(p), text)
                       or re.search(r"\braise\b|\bisinstance\b", text))
            if not guarded:
                hits.append(eq[0])
    return hits


def _check_unreachable(lines):
    """Statement directly after return/raise/break/continue in the same block."""
    hits = []
    for i, line in enumerate(lines):
        if not re.match(r"^\s*(?:return|raise|break|continue)\b", line):
            continue
        if line.rstrip().endswith(":"):
            continue
        if any(line.count(a) > line.count(b) for a, b in ("()", "[]", "{}")):
            continue  # statement continues on the next line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            continue
        nxt = lines[j]
        if (_indent(nxt) == _indent(line)
                and not re.match(r"\s*(?:else|elif|except|finally|case)\b", nxt)
                and not re.match(r"\s*[)\]}]", nxt)):
            hits.append(j)
    return hits


def _check_sql_missing_where(lines):
    """UPDATE or DELETE statement with no WHERE clause (changes every row)."""
    text = "\n".join(lines)
    hits, pos = [], 0
    for stmt in text.split(";"):
        m = re.match(r"\s*(UPDATE|DELETE)\b", stmt, re.I)
        if m and not re.search(r"\bWHERE\b", stmt, re.I):
            hits.append(text.count("\n", 0, pos + m.start(1)))
        pos += len(stmt) + 1
    return hits


# ──────────────────────────────────────────────────────────────────────
# Rule table
# ──────────────────────────────────────────────────────────────────────

def _secret_rx(names):
    """Hardcoded credential: NAME = "literal"  or  "name": "literal"."""
    return (r"(?<![\"'\w])\w*(?:%s)\w*\s*(?:\[\s*\d*\s*\])?\s*[:=]\s*[\"'][^\"'\n]+[\"']"
            r"|[\"'][\w-]*(?:%s)[\w-]*[\"']\s*:\s*[\"'][^\"'\n]+[\"']" % (names, names))


_SQL_PREFIX = r"(?:select\b.*?\bfrom\b|insert\s+into\b|update\s+\w+\s+set\b|delete\s+from\b)"

RULES = []


def _rule(rid, severity, category, message, fix, rx=None, check=None, view="masked",
          multiline=False, flags=0, in_loop=False, needs_no_exit=False):
    """Register a rule. `rx` / `check` is a single value (all of `langs`) or a
    {language: value} dict."""
    compiled, checks = {}, {}
    if rx is not None:
        for lang, pattern in rx.items():
            f = flags | (re.MULTILINE if multiline else 0)
            compiled[lang] = re.compile(pattern, f)
    if check is not None:
        checks = dict(check)
    RULES.append({
        "id": rid, "severity": severity, "category": category,
        "message": message, "fix": fix,
        "rx": compiled, "check": checks, "view": view,
        "multiline": multiline, "in_loop": in_loop, "needs_no_exit": needs_no_exit,
        "languages": tuple(sorted(set(compiled) | set(checks))),
    })


def _all(pattern, langs):
    return {lang: pattern for lang in langs}


# ── Security (HIGH) ──────────────────────────────────────────────────
_rule(1, "HIGH", "Security",
      "eval() is dangerous — can execute any code injected by user!",
      "Remove eval(). Use ast.literal_eval() in Python or JSON.parse() in JavaScript.",
      rx=_all(r"(?<![\w.])eval\s*\(", (PY, JS)))
_rule(2, "HIGH", "Security",
      "exec() is a serious security risk — executes arbitrary code!",
      "Remove exec(). Use proper function calls or logic instead.",
      rx={PY: r"(?<![\w.])exec\s*\("})
_rule(3, "HIGH", "Security",
      "Hardcoded password detected — never store passwords in code!",
      "Use environment variables: import os; password = os.environ.get('PASSWORD')",
      rx=_all(_secret_rx(r"password|passwd|pwd"), ALL_LANGS), view="code", flags=re.I)
_rule(4, "HIGH", "Security",
      "Hardcoded secret detected — store secrets in environment variables!",
      "Use: import os; secret = os.environ.get('SECRET_KEY')",
      rx=_all(_secret_rx(r"secret|private_?key"), ALL_LANGS), view="code", flags=re.I)
_rule(5, "HIGH", "Security",
      "Hardcoded API key or token in code — this is a security vulnerability!",
      "Use: api_key = os.environ.get('API_KEY') and keep the value in a .env file",
      rx=_all(_secret_rx(r"api[_-]?key|access_?token|auth_?token|bearer_?token"), ALL_LANGS),
      view="code", flags=re.I)
_rule(6, "HIGH", "Security",
      "pickle.load() can execute malicious code from untrusted files!",
      "Use json.load() for safe data loading instead of pickle.",
      rx={PY: r"\bpickle\.loads?\s*\("})
_rule(7, "HIGH", "Security",
      "subprocess with shell=True allows command injection attacks!",
      "Use shell=False and pass the command as a list: subprocess.run(['ls', '-la'])",
      rx={PY: r"\bshell\s*=\s*True\b"})
_rule(8, "HIGH", "Security",
      "Dynamic import with __import__() can be a security risk!",
      "Use importlib.import_module() or regular static imports instead.",
      rx={PY: r"(?<![\w.])__import__\s*\("})

# ── Common bugs ──────────────────────────────────────────────────────
_rule(9, "MEDIUM", "Bug",
      "Bare except: catches everything including system exits — too broad!",
      "Catch specific exceptions: except ValueError: or except TypeError:",
      rx={PY: r"^\s*except\s*:"})
_rule(10, "LOW", "Bug",
      "Too broad exception — catches almost every error, hides real bugs",
      "Catch the specific exception you expect, e.g. except ZeroDivisionError:",
      rx={PY: r"^\s*except\s+Exception\s*(?:as\s+\w+)?\s*:"})
_rule(11, "MEDIUM", "Bug",
      "File opened without 'with' — file may not close properly on error!",
      "Use: with open('file.txt', 'r') as f:  — this closes the file automatically",
      rx={PY: r"^(?!\s*(?:async\s+)?(?:with|def)\b).*?(?<![\w.])open\s*\("})
_rule(12, "MEDIUM", "Best Practice",
      "Global variable usage — makes code hard to test and debug",
      "Pass the variable as a function parameter instead of using global.",
      rx={PY: r"^\s*global\s+\w"})
_rule(13, "MEDIUM", "Bug",
      "Infinite loop with no break, return or raise — it will never stop!",
      "Add a break condition or use a proper loop condition: while count < limit:",
      rx={PY: r"^\s*while\s+(?:True|1)\s*:"}, needs_no_exit=True)
_rule(14, "MEDIUM", "Performance",
      "time.sleep() blocks the entire program — bad for performance!",
      "Consider async/await or threading for non-blocking delays.",
      rx={PY: r"\btime\.sleep\s*\("})
_rule(15, "MEDIUM", "Bug",
      "assert is disabled in optimized Python (-O flag) — not safe for validation!",
      "Use: if not condition: raise ValueError('message') instead of assert",
      rx={PY: r"^\s*assert\b"})
_rule(16, "MEDIUM", "Best Practice",
      "Wildcard import pollutes the namespace and causes name conflicts!",
      "Import only what you need: from math import sqrt, pi",
      rx={PY: r"^\s*from\s+[\w.]+\s+import\s+\*",
          JAVA: r"^\s*import\s+(?:static\s+)?[\w.]+\.\*\s*;"})
_rule(17, "MEDIUM", "Security",
      "User input used directly — always validate before processing!",
      "Validate input: val = input('Enter: '); if not val.isdigit(): raise ValueError",
      rx={PY: r"(?<![\w.])(?<!int\()(?<!float\()input\s*\("})
_rule(18, "HIGH", "Bug",
      "Division by zero detected — this will crash your program!",
      "Add a check: if divisor != 0: result = num / divisor",
      rx=_all(r"[\w)\]]\s*//?\s*0(?:\.0*)?(?![\w.])", ALL_LANGS))
_rule(19, "MEDIUM", "Bug",
      "Direct int(input()) will crash if the user types a non-number!",
      "Wrap in try-except: try: val = int(input('Enter: ')) except ValueError: print('Not a number!')",
      rx={PY: r"\bint\s*\(\s*input\s*\("})
_rule(20, "MEDIUM", "Bug",
      "Direct float(input()) crashes on non-numeric input!",
      "Wrap in try-except: try: val = float(input()) except ValueError: ...",
      rx={PY: r"\bfloat\s*\(\s*input\s*\("})

# ── Best practices / code quality ────────────────────────────────────
_rule(21, "LOW", "Best Practice",
      "Comparing to None with ==/!= — use an identity check instead",
      "Replace 'x == None' with 'x is None' (and '!=' with 'is not')",
      rx={PY: r"[=!]=\s*None\b"})
_rule(22, "LOW", "Best Practice",
      "Comparing to True with ==/!= is redundant — use the boolean value directly",
      "Replace 'if x == True:' with 'if x:'",
      rx={PY: r"[=!]=\s*True\b"})
_rule(23, "LOW", "Best Practice",
      "Comparing to False with ==/!= is redundant — use 'not' instead",
      "Replace 'if x == False:' with 'if not x:'",
      rx={PY: r"[=!]=\s*False\b"})
_rule(24, "LOW", "Best Practice",
      "print() in code — use the logging module for production output",
      "Use: import logging; logging.info('message') instead of print()",
      rx={PY: r"(?<![\w.])print\s*\("})
_rule(25, "LOW", "Code Quality",
      "TODO/FIXME comment found — incomplete code!",
      "Complete the TODO before submitting or deploying code.",
      rx=_all(r"\b(?:TODO|FIXME|XXX)\b", ALL_LANGS), view="comments")
_rule(26, "LOW", "Code Quality",
      "Empty 'pass' block — is this intentional? Empty blocks can hide bugs",
      "Add proper logic or raise NotImplementedError('Not implemented yet')",
      rx={PY: r"(?:^|:)\s*pass\s*$"})
_rule(27, "LOW", "Best Practice",
      "range(len(list)) is not Pythonic — use enumerate() instead",
      "Replace with: for i, item in enumerate(my_list):",
      rx={PY: r"\brange\s*\(\s*len\s*\("})
_rule(28, "LOW", "Best Practice",
      "Use the shorthand operator — x += 1 is cleaner",
      "Replace 'x = x + 1' with 'x += 1'",
      rx=_all(r"\b([A-Za-z_]\w*)\s*=\s*\1\s*\+\s*\d+\s*;?\s*$", CODE_LANGS))
_rule(29, "LOW", "Best Practice",
      "Avoid 'if len(x) > 0' — in Python just use 'if x:'",
      "Replace 'if len(my_list) > 0:' with 'if my_list:'",
      rx={PY: r"^\s*(?:el)?if\s+(?:not\s+)?len\s*\(\s*\w+\s*\)\s*(?:>\s*0|==\s*0|!=\s*0|>=\s*1)\s*:"})
_rule(30, "LOW", "Best Practice",
      "Using type() for a type check — use isinstance() instead",
      "Replace 'type(x) == int' with 'isinstance(x, int)' — it supports inheritance too",
      rx={PY: r"\btype\s*\([^)]*\)\s*[=!]="})

# ── Performance ──────────────────────────────────────────────────────
_rule(31, "LOW", "Performance",
      "Building strings with + str(...) is clumsy and slow — use f-strings",
      "Use an f-string: f'Hello {name}' instead of 'Hello ' + str(name)",
      rx={PY: r"\+\s*str\s*\(|\bstr\s*\([^()]*\)\s*\+"})
_rule(32, "LOW", "Performance",
      "append() inside a loop — a list comprehension is usually faster and shorter",
      "Use a list comprehension: result = [process(x) for x in items]",
      rx={PY: r"\.append\s*\("}, in_loop=True)
_rule(33, "LOW", "Performance",
      "Membership test on range() is O(n) — a direct comparison is O(1)",
      "Replace 'x in range(0, 100)' with '0 <= x < 100'",
      rx={PY: r"\bnot\s+in\s+range\s*\(|^\s*(?:el)?if\b[^:]*\bin\s+range\s*\("})
_rule(34, "MEDIUM", "Performance",
      "Double sorting detected — unnecessary duplicate operation!",
      "Sort only once with the right key: sorted(data, key=lambda x: x['name'])",
      rx={PY: r"\bsorted\s*\(\s*sorted\s*\("})
_rule(35, "LOW", "Performance",
      "dict() constructor is slower than the {} literal",
      "Replace dict() with {} for an empty dict.",
      rx={PY: r"(?<![\w.])dict\s*\(\s*\)"})

# ── Student mistakes ─────────────────────────────────────────────────
_rule(36, "LOW", "Code Quality",
      "Generic function name 'function' — use a descriptive name!",
      "Name functions by what they do: def calculate_sum() or def get_user_data()",
      rx={PY: r"\bdef\s+function\s*\("})
_rule(37, "HIGH", "Bug",
      "Index out of range! The last valid index is len-1, not len",
      "Use lst[-1] (Python) or arr[arr.length - 1] to get the last element safely",
      rx={PY: r"\b(\w+)\s*\[\s*len\s*\(\s*\1\s*\)\s*\]",
          JS: r"\b(\w+)\s*\[\s*\1\.length\s*\]",
          JAVA: r"\b(\w+)\s*\[\s*\1\.length\s*\]"})
_rule(38, "HIGH", "Bug",
      "Double self.self — this is a logic error in a class!",
      "Use just self.attribute, not self.self.attribute",
      rx={PY: r"\bself\.self\."})
_rule(39, "LOW", "Bug",
      "Recursion stops only at n == 0 — a negative input recurses forever (RecursionError)",
      "Also handle n < 0: raise ValueError for negative input before recursing",
      check={PY: _check_missing_negative_guard})
_rule(40, "MEDIUM", "Bug",
      "Variable name shadows a Python built-in function (sum, max, min, len, input, id, type, ...)",
      "Use a more specific name: total, largest, smallest, item_count, ...",
      rx={PY: r"^\s*(?:sum|max|min|len|input|id|type|sorted|map|filter|any|all|abs|round|next|iter|format|zip|range)\s*=(?!=)"})
_rule(41, "MEDIUM", "Bug",
      "Variable name shadows a Python built-in type (dict, set, tuple, int, float, bool, ...)",
      "Use a specific name: user_dict, id_set, point_tuple, count, ...",
      rx={PY: r"^\s*(?:dict|set|tuple|int|float|bool|bytes|frozenset)\s*=(?!=)"})
_rule(42, "HIGH", "Bug",
      "Shadowing built-in 'str' — breaks every str() call in this scope!",
      "Use a different variable name: name_str, text_value, etc.",
      rx={PY: r"^\s*str\s*=(?!=)"})
_rule(43, "HIGH", "Bug",
      "Shadowing built-in 'list' — breaks every list() call in this scope!",
      "Use a different variable name: my_items, data_list, etc.",
      rx={PY: r"^\s*list\s*=(?!=)"})
_rule(44, "HIGH", "Bug",
      "Incorrect __init__ signature — self should not appear twice!",
      "Correct form: def __init__(self, param1, param2):",
      rx={PY: r"\bdef\s+__init__\s*\(\s*self\s*,\s*self\b"})
_rule(45, "MEDIUM", "Bug",
      "'elif True:' is always true — use 'else:' instead",
      "Replace 'elif True:' with 'else:'",
      rx={PY: r"^\s*elif\s+True\s*:"})

# ── Logic errors ─────────────────────────────────────────────────────
_rule(46, "HIGH", "Bug",
      "Assignment '=' inside an if condition — use '==' for comparison!",
      "Change 'if x = value' to 'if x == value'",
      rx=_all(r"\bif\s*\(?\s*[\w.\[\]]+\s*=(?![=>])", CODE_LANGS))
_rule(47, "MEDIUM", "Bug",
      "Silently continuing after an exception — the error is hidden from the user!",
      "Log the error before continuing: logging.warning(f'Skipping: {e}')",
      rx={PY: r"^[ \t]*(?P<at>except\b[^\n:]*):\s*continue\b"}, multiline=True)
_rule(48, "LOW", "Best Practice",
      "Multiple imports on one line — put each on its own line (PEP 8)",
      "One import per line: import os (newline) import sys",
      rx={PY: r"^\s*import\s+[\w.]+\s*,\s*[\w.]+"})
_rule(49, "LOW", "Best Practice",
      "Returning True/False separately — return the condition directly",
      "Simplify: return condition  instead of  if condition: return True / return False",
      rx={PY: r"^[ \t]*(?P<at>return\s+True)[ \t]*\n\s*(?:else\s*:\s*)?return\s+False\b"},
      multiline=True)
_rule(50, "MEDIUM", "Bug",
      "Exception silently swallowed with 'pass' — errors disappear without a trace!",
      "Handle or log the error: except ValueError as e: logging.error(e)",
      rx={PY: r"^[ \t]*(?P<at>except\b[^\n:]*):\s*pass\b"}, multiline=True)

# ── Added per language (so every supported language has its own rules) ─
_rule(51, "MEDIUM", "Bug",
      "Empty catch block — the error is swallowed and nobody will ever know!",
      "Log or handle the exception inside the catch block.",
      rx=_all(r"\bcatch\s*(?:\([^)]*\))?\s*\{\s*\}", (JS, JAVA, CPP)), multiline=True)
_rule(52, "LOW", "Best Practice",
      "'var' has function scope and hoisting surprises — use let or const",
      "Replace 'var' with 'const' (or 'let' if the value changes).",
      rx={JS: r"^\s*var\s+\w"})
_rule(53, "LOW", "Best Practice",
      "Loose equality (==, !=) performs type coercion — use strict equality",
      "Use === and !== instead of == and !=.",
      rx={JS: r"(?<![=!<>])(?:==|!=)(?!=)"})
_rule(54, "LOW", "Best Practice",
      "console.log() left in code — remove it or use a proper logger",
      "Remove debug output before shipping, or use a logging library.",
      rx={JS: r"\bconsole\.(?:log|debug)\s*\("})
_rule(55, "HIGH", "Security",
      "Assigning to innerHTML can inject scripts (XSS)!",
      "Use textContent for plain text, or sanitize the HTML first.",
      rx={JS: r"\.(?:inner|outer)HTML\s*\+?="})
_rule(56, "MEDIUM", "Security",
      "document.write() is unsafe and blocks page rendering",
      "Build elements with the DOM API (createElement / appendChild).",
      rx={JS: r"\bdocument\.write(?:ln)?\s*\("})
_rule(57, "LOW", "Best Practice",
      "System.out.print in code — use a logger instead",
      "Use java.util.logging or SLF4J: logger.info(...)",
      rx={JAVA: r"\bSystem\.out\.print(?:ln|f)?\s*\("})
_rule(58, "LOW", "Best Practice",
      "printStackTrace() writes to stderr and hides context — log the exception",
      "Use logger.error(\"message\", e) instead.",
      rx={JAVA: r"\.printStackTrace\s*\(\s*\)"})
_rule(59, "LOW", "Bug",
      "Catching Exception/Throwable (or ...) is too broad — it hides real bugs",
      "Catch the specific exception types you can actually handle.",
      rx={JAVA: r"\bcatch\s*\(\s*(?:final\s+)?(?:Exception|Throwable)\s+\w+\s*\)",
          CPP: r"\bcatch\s*\(\s*\.\.\.\s*\)"})
_rule(60, "LOW", "Best Practice",
      "'using namespace std;' pollutes the global namespace",
      "Qualify names (std::cout) or import only what you need.",
      rx={CPP: r"\busing\s+namespace\s+std\s*;"})
_rule(61, "HIGH", "Security",
      "Unsafe C string function — no bounds checking, classic buffer overflow!",
      "Use fgets / strncpy / snprintf, or std::string.",
      rx={CPP: r"(?<![\w.])(?:gets|strcpy|strcat|sprintf)\s*\("})
_rule(62, "HIGH", "Security",
      "Running shell commands from code allows command injection!",
      "Avoid the shell: use subprocess with a list of arguments (Python) or a safe API.",
      rx={PY: r"\bos\.system\s*\(", CPP: r"(?<![\w.:>])system\s*\("})
_rule(63, "MEDIUM", "Best Practice",
      "Manual memory management (malloc / new[]) is easy to leak — use std::vector or smart pointers",
      "Use std::vector<T> or std::unique_ptr<T[]> so memory is freed automatically.",
      rx={CPP: r"(?<![\w.])(?:malloc|calloc)\s*\(|\bnew\s+[\w:]+\s*\["})
_rule(64, "LOW", "Performance",
      "SELECT * reads every column — list only the columns you need",
      "Write the column names: SELECT id, name FROM ...",
      rx={SQL: r"\bSELECT\s+(?:DISTINCT\s+)?\*"}, flags=re.I)
_rule(65, "HIGH", "Bug",
      "Destructive statement (DROP) — this permanently deletes data!",
      "Back up first, run in a transaction, and never run DROP from application code.",
      rx={SQL: r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b"}, flags=re.I)
_rule(66, "HIGH", "Bug",
      "UPDATE/DELETE without a WHERE clause changes every row in the table!",
      "Add a WHERE clause (and test it with a SELECT first).",
      check={SQL: _check_sql_missing_where})
_rule(67, "HIGH", "Security",
      "SQL query built by joining strings — vulnerable to SQL injection!",
      "Use parameterized queries: cursor.execute('... WHERE id = %s', (user_id,))",
      rx={PY: r"\"\s*" + _SQL_PREFIX + r"[^\"\n]*\"\s*(?:\+|%|\.format\s*\()"
              r"|'\s*" + _SQL_PREFIX + r"[^'\n]*'\s*(?:\+|%|\.format\s*\()"
              r"|\bf[\"']\s*" + _SQL_PREFIX + r"[^\n]*\{",
          JS: r"[\"']\s*" + _SQL_PREFIX + r"[^\"'\n]*[\"']\s*\+"
              r"|`\s*" + _SQL_PREFIX + r"[^`\n]*\$\{",
          JAVA: r"\"\s*" + _SQL_PREFIX + r"[^\"\n]*\"\s*\+"},
      view="code", flags=re.I)
_rule(68, "MEDIUM", "Bug",
      "Mutable default argument — the same list/dict is shared between calls!",
      "Use None as the default and create the object inside the function.",
      rx={PY: r"\bdef\s+\w+\s*\([^)]*=\s*(?:\[\s*\]|\{\s*\}|set\s*\(\s*\)|list\s*\(\s*\)|dict\s*\(\s*\))"})
_rule(69, "LOW", "Bug",
      "Unreachable code — nothing after return/raise/break/continue can run",
      "Delete the dead lines or move them before the return.",
      check={PY: _check_unreachable})


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def rules_for(language):
    lang = normalize_language(language) or language
    return [r for r in RULES if lang in r["languages"]]


def rule_counts():
    """{'total': N, 'python': n, ...} — used by the README and the tests."""
    counts = {"total": len(RULES)}
    for lang in ALL_LANGS:
        counts[lang] = len(rules_for(lang))
    return counts


def _bug(rule, line_no, original_lines):
    return {
        "rule_id": rule["id"],
        "line": line_no,
        "code": original_lines[line_no - 1].strip() if line_no <= len(original_lines) else "",
        "severity": rule["severity"],
        "message": rule["message"],
        "category": rule["category"],
        "fix": rule["fix"],
    }


def detect_bugs(code, language="Python"):
    """Run every rule that applies to `language` and return a list of findings.

    Raises ValueError for an unsupported language.
    """
    lang = normalize_language(language)
    if lang is None:
        raise ValueError("Unsupported language: %r" % (language,))

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code_view, masked_view, comment_view = scan(code, lang)
    full = {"code": code_view, "masked": masked_view, "comments": comment_view}
    split = {k: v.split("\n") for k, v in full.items()}
    original = code.split("\n")
    masked_lines = split["masked"]

    found = {}  # (line, rule id) -> finding, so a rule reports a line once

    for rule in RULES:
        if lang not in rule["languages"]:
            continue

        if lang in rule["check"]:
            for idx in rule["check"][lang](masked_lines):
                found.setdefault((idx + 1, rule["id"]), _bug(rule, idx + 1, original))
            continue

        rx = rule["rx"][lang]
        if rule["multiline"]:
            text = full[rule["view"]]
            for m in rx.finditer(text):
                start = m.start("at") if "at" in rx.groupindex else m.start()
                line_no = text.count("\n", 0, start) + 1
                found.setdefault((line_no, rule["id"]), _bug(rule, line_no, original))
            continue

        lines = split[rule["view"]]
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            if rule["in_loop"] and not _in_loop(masked_lines, i):
                continue
            if rule["needs_no_exit"] and _loop_has_exit(masked_lines, i):
                continue
            found.setdefault((i + 1, rule["id"]), _bug(rule, i + 1, original))

    return [found[k] for k in sorted(found)]


def get_bug_summary(bugs):
    return {
        "total": len(bugs),
        "high": len([b for b in bugs if b["severity"] == "HIGH"]),
        "medium": len([b for b in bugs if b["severity"] == "MEDIUM"]),
        "low": len([b for b in bugs if b["severity"] == "LOW"]),
    }
