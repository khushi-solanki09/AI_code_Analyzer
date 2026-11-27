"""Structural Python checks built on the `ast` module.

Each check is a function `check(ctx, **params) -> list[int]` returning the
line numbers where the problem is. Which checks run, and with which
parameters, is decided by the rule configuration (ml/rules/rules.yaml).

Unlike the regex fallback, these checks understand the code: they resolve
import aliases (`import pickle as p; p.load(f)` is still pickle.load), know
which `break` belongs to which loop, see multi-line statements, and never
look inside strings or comments.
"""

import ast
import builtins
import re

CHECKS = {}

_SQL_RX = re.compile(
    r"select\b.*?\bfrom\b|insert\s+into\b|update\s+\w+\s+set\b|delete\s+from\b", re.I | re.S)
_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
_LOOP = (ast.For, ast.AsyncFor, ast.While)


def check(name):
    def register(fn):
        CHECKS[name] = fn
        return fn
    return register


class Context:
    """Parsed module plus the lookups the checks share."""

    def __init__(self, tree):
        self.tree = tree
        self.parent = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
        self.aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.aliases[a.asname] = a.name
                    else:
                        root = a.name.split(".")[0]
                        self.aliases[root] = root
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for a in node.names:
                    if a.name != "*":
                        self.aliases[a.asname or a.name] = node.module + "." + a.name

    def qualname(self, node):
        """Dotted name of a Name/Attribute chain with import aliases resolved."""
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self.qualname(node.value)
            return base + "." + node.attr if base else None
        return None

    def call_name(self, call):
        return self.qualname(call.func) if isinstance(call, ast.Call) else None


def _stmt_lists(tree):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                yield value


def _const(node, kind=None):
    return isinstance(node, ast.Constant) and (kind is None or isinstance(node.value, kind))


def _is_number(node, value=None):
    return (_const(node) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool) and (value is None or node.value == value))


def _function_scope_nodes(fn):
    """Walk a function body without descending into nested functions/classes."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _FUNC + (ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


# ── generic call matcher ─────────────────────────────────────────────

@check("calls")
def calls(ctx, names, keywords=None, parent_names=None, not_parent_names=None, no_args=False):
    names = set(names)
    hits = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Call) or ctx.call_name(node) not in names:
            continue
        if keywords and not all(
                any(k.arg == key and _const(k.value) and k.value.value == val for k in node.keywords)
                for key, val in keywords.items()):
            continue
        parent = ctx.parent.get(node)
        wrapper = ctx.call_name(parent) if isinstance(parent, ast.Call) and node in parent.args else None
        if parent_names is not None and wrapper not in parent_names:
            continue
        if not_parent_names is not None and wrapper in not_parent_names:
            continue
        if no_args and (node.args or node.keywords):
            continue
        hits.append(node.lineno)
    return hits


# ── exceptions ───────────────────────────────────────────────────────

@check("bare_except")
def bare_except(ctx):
    return [n.lineno for n in ast.walk(ctx.tree) if isinstance(n, ast.ExceptHandler) and n.type is None]


@check("broad_except")
def broad_except(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.ExceptHandler) and isinstance(n.type, ast.Name) and n.type.id == "Exception"]


@check("except_body")
def except_body(ctx, statement):
    kind = {"pass": ast.Pass, "continue": ast.Continue}[statement]
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.ExceptHandler) and len(n.body) == 1 and isinstance(n.body[0], kind)]


# ── statements ───────────────────────────────────────────────────────

@check("open_without_with")
def open_without_with(ctx):
    hits = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Call) and ctx.call_name(node) in ("open", "io.open", "codecs.open"):
            if not isinstance(ctx.parent.get(node), ast.withitem):
                hits.append(node.lineno)
    return hits


@check("global_statement")
def global_statement(ctx):
    return [n.lineno for n in ast.walk(ctx.tree) if isinstance(n, ast.Global)]


def _can_exit(ctx, stmts, in_inner_loop):
    for s in stmts:
        if isinstance(s, _FUNC + (ast.ClassDef,)):
            continue
        if isinstance(s, (ast.Return, ast.Raise)):
            return True
        if isinstance(s, ast.Break) and not in_inner_loop:
            return True
        if isinstance(s, ast.Expr) and ctx.call_name(s.value) in ("sys.exit", "exit", "quit", "os._exit"):
            return True
        if isinstance(s, _LOOP):
            if _can_exit(ctx, s.body, True) or _can_exit(ctx, s.orelse, in_inner_loop):
                return True
            continue
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(s, field, None)
            if isinstance(sub, list) and _can_exit(ctx, sub, in_inner_loop):
                return True
        for handler in getattr(s, "handlers", []):
            if _can_exit(ctx, handler.body, in_inner_loop):
                return True
        for case in getattr(s, "cases", []):
            if _can_exit(ctx, case.body, in_inner_loop):
                return True
    return False


@check("infinite_loop")
def infinite_loop(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.While) and _const(n.test) and n.test.value
            and not _can_exit(ctx, n.body, False)]


@check("assert_statement")
def assert_statement(ctx):
    return [n.lineno for n in ast.walk(ctx.tree) if isinstance(n, ast.Assert)]


@check("assert_tuple")
def assert_tuple(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Assert) and isinstance(n.test, ast.Tuple) and n.test.elts]


@check("wildcard_import")
def wildcard_import(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)]


@check("multi_import")
def multi_import(ctx):
    return [n.lineno for n in ast.walk(ctx.tree) if isinstance(n, ast.Import) and len(n.names) > 1]


@check("empty_block")
def empty_block(ctx):
    return [block[0].lineno for block in _stmt_lists(ctx.tree)
            if len(block) == 1 and isinstance(block[0], ast.Pass)]


@check("elif_true")
def elif_true(ctx):
    return [n.orelse[0].lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.If) and len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If)
            and _const(n.orelse[0].test) and n.orelse[0].test.value is True]


@check("return_bool")
def return_bool(ctx):
    def returns(node, value):
        return isinstance(node, ast.Return) and _const(node.value) and node.value.value is value

    hits = []
    for block in _stmt_lists(ctx.tree):
        for i, s in enumerate(block):
            nxt = block[i + 1] if i + 1 < len(block) else None
            if returns(s, True) and returns(nxt, False):
                hits.append(s.lineno)
            elif (isinstance(s, ast.If) and len(s.body) == 1 and returns(s.body[0], True)
                  and ((len(s.orelse) == 1 and returns(s.orelse[0], False))
                       or (not s.orelse and returns(nxt, False)))):
                hits.append(s.body[0].lineno)
    return hits


@check("unreachable")
def unreachable(ctx):
    hits = []
    for block in _stmt_lists(ctx.tree):
        for i, s in enumerate(block[:-1]):
            if isinstance(s, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                hits.append(block[i + 1].lineno)
                break
    return hits


# ── expressions ──────────────────────────────────────────────────────

@check("divide_by_zero")
def divide_by_zero(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod))
            and _is_number(n.right, 0)]


@check("compare_constant")
def compare_constant(ctx, value):
    wanted = {"none": None, "true": True, "false": False}[value]
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.Compare) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in n.ops):
            if any(_const(x) and x.value is wanted for x in [n.left] + n.comparators):
                hits.append(n.lineno)
    return hits


@check("is_literal")
def is_literal(ctx):
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.Compare) and any(isinstance(op, (ast.Is, ast.IsNot)) for op in n.ops):
            if any(_const(x, (str, bytes, int, float)) and not isinstance(x.value, bool)
                   for x in [n.left] + n.comparators):
                hits.append(n.lineno)
    return hits


@check("range_len")
def range_len(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Call) and ctx.call_name(n) == "range"
            and any(ctx.call_name(a) == "len" for a in n.args)]


@check("shorthand_operator")
def shorthand_operator(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, (ast.Add, ast.Sub))
            and isinstance(n.value.left, ast.Name) and n.value.left.id == n.targets[0].id
            and _is_number(n.value.right)]


@check("len_compare")
def len_compare(ctx):
    hits = []
    for n in ast.walk(ctx.tree):
        if not isinstance(n, (ast.If, ast.While)):
            continue
        test = n.test.operand if isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not) else n.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and ctx.call_name(test.left) == "len"
                and len(test.left.args) == 1 and _is_number(test.comparators[0])):
            continue
        op, k = test.ops[0], test.comparators[0].value
        if (k == 0 and isinstance(op, (ast.Gt, ast.NotEq, ast.Eq))) or \
                (k == 1 and isinstance(op, (ast.GtE, ast.Lt))):
            hits.append(n.lineno)
    return hits


@check("type_compare")
def type_compare(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Compare) and ctx.call_name(n.left) == "type"
            and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in n.ops)]


@check("str_concat")
def str_concat(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
            and (ctx.call_name(n.left) == "str" or ctx.call_name(n.right) == "str")]


@check("append_in_loop")
def append_in_loop(ctx):
    hits = []
    for n in ast.walk(ctx.tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append"):
            continue
        child, node = n, ctx.parent.get(n)
        while node is not None and not isinstance(node, _FUNC + (ast.Lambda, ast.ClassDef, ast.Module)):
            in_body = isinstance(node, _LOOP) and (child in node.body or child in node.orelse)
            if in_body:
                hits.append(n.lineno)
                break
            child, node = node, ctx.parent.get(node)
    return hits


@check("range_membership")
def range_membership(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops)
            and any(ctx.call_name(c) == "range" for c in n.comparators)]


@check("nested_sorted")
def nested_sorted(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Call) and ctx.call_name(n) == "sorted"
            and n.args and ctx.call_name(n.args[0]) == "sorted"]


@check("function_name")
def function_name(ctx, name):
    return [n.lineno for n in ast.walk(ctx.tree) if isinstance(n, _FUNC) and n.name == name]


@check("index_len")
def index_len(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and ctx.call_name(n.slice) == "len" and len(n.slice.args) == 1
            and isinstance(n.slice.args[0], ast.Name) and n.slice.args[0].id == n.value.id]


@check("self_self")
def self_self(ctx):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute)
            and n.value.attr == "self" and isinstance(n.value.value, ast.Name) and n.value.value.id == "self"]


@check("shadow_builtin")
def shadow_builtin(ctx, names):
    names = set(names)
    hits = []
    for n in ast.walk(ctx.tree):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets = [n.target]
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            targets = [n.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id in names and t.id in dir(builtins):
                hits.append(n.lineno)
    return hits


@check("missing_negative_guard")
def missing_negative_guard(ctx):
    """Recursive function whose only base case is `n == 0`."""
    hits = []
    for fn in ast.walk(ctx.tree):
        if not isinstance(fn, _FUNC):
            continue
        recursive = any(
            isinstance(c, ast.Call)
            and ((isinstance(c.func, ast.Name) and c.func.id == fn.name)
                 or (isinstance(c.func, ast.Attribute) and c.func.attr == fn.name
                     and isinstance(c.func.value, ast.Name) and c.func.value.id in ("self", "cls")))
            for c in ast.walk(fn))
        if not recursive:
            continue
        body_nodes = list(ast.walk(fn))
        guarded = any(isinstance(n, ast.Raise) for n in body_nodes) or any(
            isinstance(n, ast.Call) and ctx.call_name(n) == "isinstance" for n in body_nodes)
        for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
            if arg.arg in ("self", "cls"):
                continue
            eq_zero, ordered = [], False
            for n in body_nodes:
                if not isinstance(n, ast.Compare):
                    continue
                sides = [n.left] + n.comparators
                if not any(isinstance(s, ast.Name) and s.id == arg.arg for s in sides):
                    continue
                if any(isinstance(op, ast.Eq) for op in n.ops) and any(_is_number(s, 0) for s in sides):
                    eq_zero.append(n.lineno)
                if any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in n.ops) \
                        and any(_is_number(s) for s in sides):
                    ordered = True
            if eq_zero and not ordered and not guarded:
                hits.append(min(eq_zero))
    return hits


@check("mutable_default")
def mutable_default(ctx):
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, _FUNC + (ast.Lambda,)):
            for d in n.args.defaults + [k for k in n.args.kw_defaults if k is not None]:
                if isinstance(d, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)) or (
                        isinstance(d, ast.Call) and ctx.call_name(d) in ("list", "dict", "set")):
                    hits.append(d.lineno)
    return hits


@check("duplicate_dict_keys")
def duplicate_dict_keys(ctx):
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.Dict):
            seen = set()
            for key in n.keys:
                if _const(key):
                    marker = (type(key.value), key.value)
                    if marker in seen:
                        hits.append(key.lineno)
                    seen.add(marker)
    return hits


# ── secrets and SQL injection ────────────────────────────────────────

def _name_of(target):
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript) and _const(target.slice, str):
        return target.slice.value
    return None


def _filled_string(node):
    return _const(node, str) and bool(node.value.strip())


@check("hardcoded_secret")
def hardcoded_secret(ctx, pattern):
    rx = re.compile(pattern)
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.Assign) and _filled_string(n.value):
            names = [_name_of(t) for t in n.targets]
            if any(name and rx.search(name) for name in names):
                hits.append(n.lineno)
        elif isinstance(n, ast.AnnAssign) and n.value is not None and _filled_string(n.value):
            name = _name_of(n.target)
            if name and rx.search(name):
                hits.append(n.lineno)
        elif isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if _const(k, str) and rx.search(k.value) and _filled_string(v):
                    hits.append(k.lineno)
        elif isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg and rx.search(kw.arg) and _filled_string(kw.value):
                    hits.append(kw.value.lineno)
    return hits


@check("sql_string_building")
def sql_string_building(ctx):
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.JoinedStr):
            text = "".join(p.value for p in n.values if _const(p, str))
            if any(isinstance(p, ast.FormattedValue) for p in n.values) and _SQL_RX.search(text):
                hits.append(n.lineno)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mod)):
            sides = [n.left, n.right]
            if any(_const(s, str) and _SQL_RX.search(s.value) for s in sides) \
                    and any(not _const(s) for s in sides):
                hits.append(n.lineno)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format"
              and _const(n.func.value, str) and _SQL_RX.search(n.func.value.value)):
            hits.append(n.lineno)
    return hits


# ── code quality (new, only possible with a real parser) ─────────────

@check("unused_import")
def unused_import(ctx):
    used = {n.id for n in ast.walk(ctx.tree) if isinstance(n, ast.Name)}
    for n in ast.walk(ctx.tree):  # names listed in __all__ count as used
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            used |= {e.value for e in ast.walk(n.value) if _const(e, str)}
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.Import):
            local = [(a.asname or a.name.split(".")[0]) for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module != "__future__":
            local = [(a.asname or a.name) for a in n.names if a.name != "*"]
        else:
            continue
        if any(name not in used for name in local):
            hits.append(n.lineno)
    return hits


@check("unused_variable")
def unused_variable(ctx):
    hits = []
    for fn in ast.walk(ctx.tree):
        if not isinstance(fn, _FUNC):
            continue
        everything = list(ast.walk(fn))
        if any(isinstance(n, ast.Name) and n.id in ("locals", "vars", "eval", "exec") for n in everything):
            continue
        declared = {name for n in everything if isinstance(n, (ast.Global, ast.Nonlocal)) for name in n.names}
        read = {n.id for n in everything if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Store)}
        read |= {n.target.id for n in everything if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)}
        for n in _function_scope_nodes(fn):
            targets = []
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                targets = [n.target]
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                targets = [n.optional_vars]
            for t in targets:
                if isinstance(t, ast.Name) and t.id not in read and t.id not in declared \
                        and not t.id.startswith("_"):
                    hits.append(t.lineno)
    return hits


@check("too_many_parameters")
def too_many_parameters(ctx, limit):
    hits = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, _FUNC):
            a = n.args
            names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs if x.arg not in ("self", "cls")]
            count = len(names) + (1 if a.vararg else 0) + (1 if a.kwarg else 0)
            if count > limit:
                hits.append(n.lineno)
    return hits


@check("long_function")
def long_function(ctx, limit):
    return [n.lineno for n in ast.walk(ctx.tree)
            if isinstance(n, _FUNC) and (n.end_lineno or n.lineno) - n.lineno + 1 > limit]
