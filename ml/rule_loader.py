"""Loads and validates the rule configuration (ml/rules/rules.yaml).

A rule is data: id, severity, category, message, fix, and per language one
or more ways to detect it:

    ast        a named structural check for Python (ml/python_ast.py)
    query      a tree-sitter query for JavaScript / Java / C++
    sql        a named sqlparse check for SQL (ml/sql_engine.py)
    regex      a regular expression (also the fallback when code cannot be parsed)
    text_check a named line-based check (the fallback for a few Python rules)

The whole file is validated when it loads, so a typo in a rule fails at
start-up with a clear message instead of silently never matching.
"""

import os
import re
from dataclasses import dataclass, field

import yaml

from ml import python_ast, sql_engine, text_checks, treesitter_engine

RULES_PATH = os.path.join(os.path.dirname(__file__), "rules", "rules.yaml")

LANGUAGES = ("python", "javascript", "java", "cpp", "sql")
SEVERITIES = ("HIGH", "MEDIUM", "LOW")
CATEGORIES = ("Security", "Bug", "Best Practice", "Performance", "Code Quality")
VIEWS = ("masked", "code", "comments")
FLAGS = {"IGNORECASE": re.IGNORECASE, "DOTALL": re.DOTALL}
IMPL_KEYS = {"ast", "query", "sql", "regex", "text_check", "generated", "view", "multiline",
             "flags", "in_loop", "needs_no_exit"}


class RuleConfigError(ValueError):
    """The rule file is invalid. The message lists every problem found."""


@dataclass
class Impl:
    ast: dict = None
    query: str = None
    sql: dict = None
    regex: "re.Pattern" = None
    text_check: str = None
    generated: bool = False
    view: str = "masked"
    multiline: bool = False
    in_loop: bool = False
    needs_no_exit: bool = False

    @property
    def structural(self):
        return bool(self.ast or self.query or self.sql)


@dataclass
class Rule:
    id: int
    severity: str
    category: str
    message: str
    fix: str
    impls: dict = field(default_factory=dict)

    @property
    def languages(self):
        return tuple(sorted(self.impls))


def _compile_regex(pattern, flag_names, multiline):
    flags = re.MULTILINE if multiline else 0
    for name in flag_names:
        flags |= FLAGS[name]
    return re.compile(pattern, flags)


def _check_impl(rid, lang, raw, problems):
    where = "rule %s (%s)" % (rid, lang)
    if not isinstance(raw, dict):
        problems.append("%s: implementation must be a mapping" % where)
        return None
    unknown = set(raw) - IMPL_KEYS
    if unknown:
        problems.append("%s: unknown keys %s" % (where, sorted(unknown)))
    if not (set(raw) & {"ast", "query", "sql", "regex", "text_check", "generated"}):
        problems.append("%s: needs one of ast, query, sql, regex, text_check" % where)

    impl = Impl(view=raw.get("view", "masked"), multiline=bool(raw.get("multiline", False)),
                in_loop=bool(raw.get("in_loop", False)), needs_no_exit=bool(raw.get("needs_no_exit", False)),
                generated=bool(raw.get("generated", False)))
    if impl.view not in VIEWS:
        problems.append("%s: view must be one of %s" % (where, VIEWS))

    if "ast" in raw:
        spec = dict(raw["ast"] or {})
        if lang != "python":
            problems.append("%s: 'ast' is only for python" % where)
        elif spec.get("check") not in python_ast.CHECKS:
            problems.append("%s: unknown ast check %r" % (where, spec.get("check")))
        else:
            impl.ast = spec
    if "sql" in raw:
        spec = dict(raw["sql"] or {})
        if lang != "sql":
            problems.append("%s: 'sql' is only for sql" % where)
        elif spec.get("check") not in sql_engine.CHECKS:
            problems.append("%s: unknown sql check %r" % (where, spec.get("check")))
        else:
            impl.sql = spec
    if "query" in raw:
        if lang not in treesitter_engine.GRAMMARS:
            problems.append("%s: 'query' is only for javascript, java, cpp" % where)
        else:
            try:
                treesitter_engine.compile_query(lang, raw["query"])
                impl.query = raw["query"]
            except Exception as exc:  # tree_sitter raises QueryError subclasses
                problems.append("%s: invalid tree-sitter query: %s" % (where, exc))
    if "text_check" in raw:
        if raw["text_check"] not in text_checks.TEXT_CHECKS:
            problems.append("%s: unknown text_check %r" % (where, raw["text_check"]))
        else:
            impl.text_check = raw["text_check"]
    if "regex" in raw:
        bad_flags = [f for f in raw.get("flags", []) if f not in FLAGS]
        if bad_flags:
            problems.append("%s: unknown flags %s" % (where, bad_flags))
        else:
            try:
                impl.regex = _compile_regex(raw["regex"], raw.get("flags", []), impl.multiline)
            except re.error as exc:
                problems.append("%s: invalid regex: %s" % (where, exc))
    return impl


def load_rules(path=RULES_PATH):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    problems = []
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise RuleConfigError("rules file must contain a 'rules' list")

    rules, seen = [], set()
    for entry in data["rules"]:
        rid = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(rid, int):
            problems.append("a rule has a missing or non-integer id: %r" % (entry,))
            continue
        if rid in seen:
            problems.append("duplicate rule id %d" % rid)
        seen.add(rid)
        if entry.get("severity") not in SEVERITIES:
            problems.append("rule %d: severity must be one of %s" % (rid, SEVERITIES))
        if entry.get("category") not in CATEGORIES:
            problems.append("rule %d: category must be one of %s" % (rid, CATEGORIES))
        for key in ("message", "fix"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                problems.append("rule %d: '%s' must be a non-empty string" % (rid, key))
        languages = entry.get("languages")
        if not isinstance(languages, dict) or not languages:
            problems.append("rule %d: 'languages' must be a non-empty mapping" % rid)
            continue
        rule = Rule(rid, entry.get("severity"), entry.get("category"), entry.get("message", ""),
                    entry.get("fix", ""))
        for lang, raw in languages.items():
            if lang not in LANGUAGES:
                problems.append("rule %d: unknown language %r" % (rid, lang))
                continue
            impl = _check_impl(rid, lang, raw, problems)
            if impl is not None:
                rule.impls[lang] = impl
        rules.append(rule)

    if problems:
        raise RuleConfigError("Invalid rule configuration:\n  - " + "\n  - ".join(problems))
    return rules
