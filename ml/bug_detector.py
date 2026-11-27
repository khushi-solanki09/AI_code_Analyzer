"""Bug and security detector.

The rules live in ml/rules/rules.yaml. For each language the detector first
tries to understand the code with a real parser:

    Python                      ast module             (ml/python_ast.py)
    JavaScript, Java, C++       tree-sitter queries    (ml/treesitter_engine.py)
    SQL                         sqlparse               (ml/sql_engine.py)

Structural rules see the actual program: they resolve imports, know which
`break` belongs to which loop, and never look inside strings or comments.

When the code does not parse (Python syntax error, tree-sitter error nodes)
the detector reports the syntax error for Python and falls back to the
regular-expression version of each rule, which works on a masked copy of the
source (comments removed, string contents blanked). Rules that only need text,
such as TODO comments, always use their regex.

Each finding says which engine produced it ("ast", "tree-sitter", "sqlparse"
or "regex") so a result can be traced.
"""

import ast
import warnings

from ml import python_ast, sql_engine, text_checks, treesitter_engine
from ml.preprocess import normalize_language, scan
from ml.rule_loader import LANGUAGES, load_rules

RULES = load_rules()
_BY_ID = {r.id: r for r in RULES}
SYNTAX_ERROR_RULE = 74


# ──────────────────────────────────────────────────────────────────────
# Rule queries
# ──────────────────────────────────────────────────────────────────────

def rules_for(language):
    lang = normalize_language(language) or language
    return [r for r in RULES if lang in r.impls]


def rule_counts():
    """Rule counts per language. `structural` counts rules that use a parser."""
    counts = {"total": len(RULES), "structural": {}}
    for lang in LANGUAGES:
        rules = rules_for(lang)
        counts[lang] = len(rules)
        counts["structural"][lang] = sum(1 for r in rules if r.impls[lang].structural)
    return counts


# ──────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────

def _finding(rule, line_no, original, engine, message=None):
    return {
        "rule_id": rule.id,
        "line": line_no,
        "code": original[line_no - 1].strip() if 1 <= line_no <= len(original) else "",
        "severity": rule.severity,
        "message": message or rule.message,
        "category": rule.category,
        "fix": rule.fix,
        "engine": engine,
    }


def _parse(lang, code, masked):
    """Return (structure, syntax_error). `structure` is None when the code
    cannot be understood by the language's parser."""
    if lang == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(code)
            return python_ast.Context(tree), None
        except (SyntaxError, ValueError) as exc:
            return None, exc
        except (RecursionError, MemoryError):
            return None, None
    if lang in treesitter_engine.GRAMMARS:
        parsed = treesitter_engine.Parsed(lang, code)
        return (None if parsed.has_error else parsed), None
    return sql_engine.Context(masked), None


def _run_structural(lang, impl, structure):
    if lang == "python":
        spec = dict(impl.ast)
        return python_ast.CHECKS[spec.pop("check")](structure, **spec), "ast"
    if lang == "sql":
        spec = dict(impl.sql)
        return sql_engine.CHECKS[spec.pop("check")](structure, **spec), "sqlparse"
    return structure.run(impl.query), "tree-sitter"


def _run_text(impl, full, split, masked_lines):
    """Regex and text-check fallbacks. Returns 1-based line numbers."""
    if impl.text_check:
        return [i + 1 for i in text_checks.TEXT_CHECKS[impl.text_check](masked_lines)]
    rx = impl.regex
    if impl.multiline:
        text = full[impl.view]
        lines = []
        for m in rx.finditer(text):
            start = m.start("at") if "at" in rx.groupindex else m.start()
            lines.append(text.count("\n", 0, start) + 1)
        return lines
    lines = []
    for i, line in enumerate(split[impl.view]):
        if not rx.search(line):
            continue
        if impl.in_loop and not text_checks.in_loop(masked_lines, i):
            continue
        if impl.needs_no_exit and text_checks.loop_has_exit(masked_lines, i):
            continue
        lines.append(i + 1)
    return lines


def detect_bugs(code, language="Python"):
    """Run every rule that applies to `language` and return the findings,
    sorted by line. Raises ValueError for an unsupported language."""
    lang = normalize_language(language)
    if lang is None:
        raise ValueError("Unsupported language: %r" % (language,))

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code_view, masked_view, comment_view = scan(code, lang)
    full = {"code": code_view, "masked": masked_view, "comments": comment_view}
    split = {k: v.split("\n") for k, v in full.items()}
    original = code.split("\n")
    masked_lines = split["masked"]

    structure, syntax_error = _parse(lang, code, masked_view)
    found = {}  # (line, rule id) -> finding; a rule reports a line once

    if syntax_error is not None:
        rule = _BY_ID[SYNTAX_ERROR_RULE]
        line = getattr(syntax_error, "lineno", None) or 1
        detail = getattr(syntax_error, "msg", None) or str(syntax_error)
        found[(line, rule.id)] = _finding(rule, line, original, "ast",
                                          "%s (%s)" % (rule.message, detail))

    for rule in RULES:
        impl = rule.impls.get(lang)
        if impl is None or impl.generated:
            continue
        lines = None
        if structure is not None and impl.structural:
            try:
                lines, engine = _run_structural(lang, impl, structure)
            except RecursionError:
                pass  # absurdly deep code: use the text version of the rule
        if lines is None:
            if impl.regex is None and not impl.text_check:
                continue
            lines, engine = _run_text(impl, full, split, masked_lines), "regex"
        for line in lines:
            found.setdefault((line, rule.id), _finding(rule, line, original, engine))

    return [found[k] for k in sorted(found)]


def get_bug_summary(bugs):
    return {
        "total": len(bugs),
        "high": len([b for b in bugs if b["severity"] == "HIGH"]),
        "medium": len([b for b in bugs if b["severity"] == "MEDIUM"]),
        "low": len([b for b in bugs if b["severity"] == "LOW"]),
    }
