"""SQL checks built on sqlparse (it splits statements and groups tokens, which
is enough to see structure like "UPDATE with no WHERE" without a database).
"""

import re

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Function, Where

CHECKS = {}


def check(name):
    def register(fn):
        CHECKS[name] = fn
        return fn
    return register


class Context:
    """Statements of the (comment-free) SQL text with their line numbers."""

    def __init__(self, masked_text):
        self.text = masked_text
        self.statements = []  # (statement, start offset)
        offset = 0
        for stmt in sqlparse.parse(masked_text):
            self.statements.append((stmt, offset))
            offset += len(str(stmt))

    def line_of(self, offset):
        return self.text.count("\n", 0, offset) + 1

    def first_word_offset(self, stmt, start):
        raw = str(stmt)
        return start + (len(raw) - len(raw.lstrip()))


@check("select_star")
def select_star(ctx):
    hits = []
    for stmt, start in ctx.statements:
        if stmt.get_type() != "SELECT":
            continue
        pos = start
        for tok in stmt.flatten():
            if tok.ttype is T.Wildcard and not _inside(tok, Function):
                hits.append(ctx.line_of(pos))
            pos += len(tok.value)
    return hits


def _inside(token, cls):
    parent = token.parent
    while parent is not None:
        if isinstance(parent, cls):
            return True
        parent = parent.parent
    return False


@check("drop_statement")
def drop_statement(ctx):
    hits = []
    for stmt, start in ctx.statements:
        words = re.findall(r"[A-Za-z]+", str(stmt), re.S)[:2]
        if len(words) == 2 and words[0].upper() == "DROP" and words[1].upper() in ("TABLE", "DATABASE", "SCHEMA"):
            hits.append(ctx.line_of(ctx.first_word_offset(stmt, start)))
    return hits


@check("missing_where")
def missing_where(ctx):
    hits = []
    for stmt, start in ctx.statements:
        if stmt.get_type() in ("UPDATE", "DELETE") and not any(isinstance(t, Where) for t in stmt.tokens):
            hits.append(ctx.line_of(ctx.first_word_offset(stmt, start)))
    return hits
