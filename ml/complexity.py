"""Time/space complexity estimation and basic code metrics.

Python is analysed with the `ast` module, so loops, recursion and calls are
understood structurally. The other languages use a brace-matching scanner
over comment- and string-free text. Both paths handle the cases the old
line-counting version got wrong:

* loops that halve or double their counter           -> O(log n), not O(n)
* recursion with memoization (lru_cache, memo dicts) -> O(n), not O(2^n)
* divide-and-conquer recursion                        -> O(log n) / O(n log n)
* library calls with hidden cost (sorted, sum, ...)   -> counted, not ignored
* loops over a constant range such as range(10)       -> O(1)

The result is a static *estimate*. It cannot know input sizes or what
imported functions cost, so treat it as a first guess.
"""

import ast
import re

from ml.preprocess import normalize_language, scan

# A complexity term is (exponential, n_power, log_power)
Term = tuple
ONE = (False, 0, 0)


def _mul(a, b):
    if a[0] or b[0]:
        return (True, 0, 0)
    return (False, a[1] + b[1], a[2] + b[2])


def _max(*terms):
    return max(terms, key=lambda t: (t[0], t[1], t[2]))


def _fmt(t):
    exp, a, b = t
    if exp:
        return "O(2^n) — Exponential"
    if a == 0 and b == 0:
        return "O(1) — Constant"
    if a == 0:
        return "O(log n) — Logarithmic" if b == 1 else "O((log n)^%d)" % b
    if b == 0:
        names = {1: "Linear", 2: "Quadratic", 3: "Cubic"}
        label = "O(n)" if a == 1 else "O(n^%d)" % a if a > 3 else "O(n%s)" % "²³"[a - 2]
        return "%s — %s" % (label, names.get(a, "Polynomial"))
    n = "n" if a == 1 else "n^%d" % a
    return "O(%s log n) — Linearithmic" % n if b == 1 else "O(%s (log n)^%d)" % (n, b)


# ──────────────────────────────────────────────────────────────────────
# Python: AST based
# ──────────────────────────────────────────────────────────────────────

_SORT_NAMES = {"sorted", "sort"}
_LINEAR_BUILTINS = {"sum", "max", "min", "any", "all", "list", "set", "tuple",
                    "dict", "reversed", "enumerate", "zip", "len_of_iter", "join",
                    "index", "count", "remove", "insert", "extend", "copy"}
_CACHE_DECORATORS = {"lru_cache", "cache", "cached_property", "memoize"}


def _call_name(node):
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_const_range(node):
    """for ... in range(<numbers only>) — a fixed number of iterations."""
    it = node.iter
    return (isinstance(it, ast.Call) and _call_name(it) == "range"
            and it.args and all(isinstance(a, ast.Constant) for a in it.args))


def _halves(expr):
    """Does this expression shrink/grow its value geometrically?"""
    for n in ast.walk(expr):
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.FloorDiv, ast.Div, ast.RShift)):
                return True
            if isinstance(n.op, (ast.Mult, ast.LShift)) and any(
                    isinstance(x, ast.Constant) and isinstance(x.value, (int, float)) and x.value >= 2
                    for x in (n.left, n.right)):
                return True
        if isinstance(n, ast.Name) and n.id in ("mid", "middle"):
            return True
    return False


def _halving_slice(a):
    """arr[:mid] or arr[len(arr)//2:] — a slice that cuts the data geometrically."""
    if isinstance(a, ast.Subscript) and isinstance(a.slice, ast.Slice):
        return any(x is not None and _halves(x) for x in (a.slice.lower, a.slice.upper))
    return False


def _while_is_logarithmic(node):
    """while-loop whose counter is halved/doubled, or a binary-search style loop."""
    for n in ast.walk(node):
        if isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.FloorDiv, ast.Div, ast.RShift,
                                                               ast.LShift)):
            return True
        if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Mult) and \
                isinstance(n.value, ast.Constant) and n.value.value not in (0, 1, -1):
            return True
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) for t in n.targets):
            tgt = {t.id for t in n.targets if isinstance(t, ast.Name)}
            if _halves(n.value) and any(isinstance(x, ast.Name) and x.id in tgt for x in ast.walk(n.value)):
                return True  # x = x // 2  /  x = x * 2
            if "mid" in {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)} and tgt & {"lo", "low", "left", "l", "hi", "high", "right", "r", "start", "end"}:
                return True  # lo = mid + 1  /  hi = mid - 1
    return False


class _PyAnalyzer:
    def __init__(self, tree):
        self.tree = tree
        self.funcs = {}
        self.max_struct_term = ONE
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[n.name] = n

    # --- cost of a statement list -------------------------------------
    def block(self, stmts, fname=None, depth=0):
        total = ONE
        for s in stmts:
            total = _max(total, self.stmt(s, fname, depth))
        return total

    def stmt(self, s, fname, depth):
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return ONE  # analysed separately
        if isinstance(s, ast.For) or isinstance(s, ast.AsyncFor):
            inner = self.block(s.body + s.orelse, fname, depth + 1)
            iters = ONE if _is_const_range(s) else (False, 1, 0)
            if not _is_const_range(s):
                self._grow(s, iters)
            return _max(_mul(iters, inner), self.expr(s.iter))
        if isinstance(s, ast.While):
            inner = self.block(s.body + s.orelse, fname, depth + 1)
            iters = (False, 0, 1) if _while_is_logarithmic(s) else (False, 1, 0)
            self._grow(s, iters)
            return _mul(iters, inner)
        if isinstance(s, ast.If):
            return _max(self.expr(s.test), self.block(s.body, fname, depth), self.block(s.orelse, fname, depth))
        if isinstance(s, ast.Try):
            parts = s.body + s.orelse + s.finalbody + [x for h in s.handlers for x in h.body]
            return self.block(parts, fname, depth)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            return self.block(s.body, fname, depth)
        total = ONE
        for child in ast.iter_child_nodes(s):
            if isinstance(child, ast.expr):
                total = _max(total, self.expr(child))
        return total

    def _grow(self, loop, iters):
        """A container that is appended to inside a loop grows with the loop."""
        for n in ast.walk(loop):
            if isinstance(n, ast.Call) and _call_name(n) in ("append", "add", "extend", "insert", "appendleft"):
                self.max_struct_term = _max(self.max_struct_term, iters)
            if isinstance(n, ast.Subscript) and isinstance(getattr(n, "ctx", None), ast.Store):
                self.max_struct_term = _max(self.max_struct_term, iters)

    # --- cost of an expression (comprehensions and library calls) ------
    def expr(self, e):
        total = ONE
        for n in ast.walk(e):
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                t = ONE
                for g in n.generators:
                    t = _mul(t, ONE if _is_const_range(ast.For(iter=g.iter, target=g.target, body=[], orelse=[])) else (False, 1, 0))
                total = _max(total, t)
                if not isinstance(n, ast.GeneratorExp):
                    self.max_struct_term = _max(self.max_struct_term, t)
            if isinstance(n, ast.Call):
                name = _call_name(n)
                if name in _SORT_NAMES:
                    total = _max(total, (False, 1, 1))
                elif name in _LINEAR_BUILTINS and n.args:
                    total = _max(total, (False, 1, 0))
        return total


def _py_recursion(an):
    """Classify each self-recursive function. Returns (time_term, space_term, calls, memo)."""
    time_t, space_t, total_calls, memo_any = ONE, ONE, 0, False
    for name, fn in an.funcs.items():
        calls = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]
        if not calls:
            continue
        total_calls += len(calls)
        decorated = any((_call_name(d) if isinstance(d, ast.Call) else
                         d.id if isinstance(d, ast.Name) else
                         d.attr if isinstance(d, ast.Attribute) else None) in _CACHE_DECORATORS
                        for d in fn.decorator_list)
        arg_names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        memo_name = bool(arg_names & {"memo", "cache", "dp", "seen", "visited", "memory"})
        local_memo = any(isinstance(n, ast.Name) and n.id in ("memo", "cache", "dp", "memory")
                         for n in ast.walk(fn))
        memoized = decorated or memo_name or local_memo
        memo_any = memo_any or memoized
        halving = any(_halves(a) or _halving_slice(a) for c in calls for a in c.args)
        # work done outside the recursive calls, in one invocation
        sub = an.block(fn.body, name)
        linear_work = sub[1] >= 1 or any(isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)
                                          for n in ast.walk(fn))
        if memoized:
            t, s = (False, 1, 0), (False, 1, 0)
        elif len(calls) >= 2 and not halving:
            t, s = (True, 0, 0), (False, 1, 0)
        elif len(calls) >= 2 and halving:
            t = (False, 1, 1) if linear_work else (False, 1, 0)
            s = (False, 0, 1)
        elif halving:
            t, s = (False, 0, 1), (False, 0, 1)
        else:
            t = (False, 2, 0) if linear_work else (False, 1, 0)
            s = (False, 1, 0)
        time_t, space_t = _max(time_t, t), _max(space_t, s)
    return time_t, space_t, total_calls, memo_any


_NEST_BLOCKS = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith,
                ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _nest(node, depth=0):
    best = depth
    for child in ast.iter_child_nodes(node):
        best = max(best, _nest(child, depth + 1 if isinstance(child, _NEST_BLOCKS) else depth))
    return best


def _is_container(value):
    if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    return isinstance(value, ast.Call) and _call_name(value) in (
        "list", "dict", "set", "defaultdict", "deque", "Counter", "OrderedDict")


def _python_metrics(code):
    tree = ast.parse(code)
    an = _PyAnalyzer(tree)

    body_cost = an.block(tree.body)
    func_cost = ONE
    for fn in an.funcs.values():
        func_cost = _max(func_cost, an.block(fn.body, fn.name))
    rec_time, rec_space, rec_calls, memo = _py_recursion(an)
    time_t = _max(body_cost, func_cost, rec_time)

    # --- space ---
    space_t = _max(rec_space, an.max_struct_term)
    for n in ast.walk(tree):
        if isinstance(n, ast.ListComp) and not all(_is_const_range(ast.For(iter=g.iter, target=g.target, body=[], orelse=[])) for g in n.generators):
            for x in ast.walk(n.elt):
                if isinstance(x, ast.ListComp) or (
                        isinstance(x, ast.BinOp) and isinstance(x.op, ast.Mult) and isinstance(x.left, ast.List)
                        and not isinstance(x.right, ast.Constant)):
                    space_t = _max(space_t, (False, 2, 0))  # [[0] * n for _ in range(n)]
                    break
        if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
                and isinstance(n.left, ast.List) and not isinstance(n.right, ast.Constant)):
            space_t = _max(space_t, (False, 1, 0))  # [0] * n

    walk = list(ast.walk(tree))
    loops = sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in walk) + \
        sum(len(n.generators) for n in walk
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)))
    return {
        "time": time_t, "space": space_t, "recursive_calls": rec_calls, "memo": memo,
        "functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for n in walk),
        "classes": sum(isinstance(n, ast.ClassDef) for n in walk),
        "loops": loops,
        "conditions": sum(isinstance(n, ast.If) for n in walk),
        "depth": _nest(tree),
        "structures": sum(isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None
                          and _is_container(n.value) for n in walk),
    }


# ──────────────────────────────────────────────────────────────────────
# Other languages: brace matching on comment/string-free text
# ──────────────────────────────────────────────────────────────────────

_LOOP_RX = re.compile(r"\b(for|while)\s*\(")
_FUNC_RX = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{}()]*(?:\([^)]*\)[^;{}()]*)*\)\s*(?:const\s*)?(?:throws\s+[\w.,\s]+)?\{")
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "sizeof", "else"}


def _match_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def _match_paren(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def _brace_metrics(masked, lang):
    loops = []  # (start, end, header, body)
    for m in _LOOP_RX.finditer(masked):
        po = m.end() - 1
        pc = _match_paren(masked, po)
        header = masked[po:pc + 1]
        j = pc + 1
        while j < len(masked) and masked[j].isspace():
            j += 1
        if j < len(masked) and masked[j] == "{":
            end = _match_brace(masked, j)
            body = masked[j:end + 1]
        else:
            end = masked.find(";", j)
            end = len(masked) if end == -1 else end
            body = masked[j:end + 1]
        loops.append((m.start(), end, header, body))

    def is_log(header, body):
        h = header + body
        return bool(re.search(r"(?:/=|>>=|<<=|\*=)\s*[2-9]|=\s*\w+\s*(?:/|\*)\s*[2-9]|>>\s*1|<<\s*1|\bmid\b", h))

    def is_const(header):
        # for (int i = 0; i < 10; i++)
        return bool(re.search(r"<=?\s*\d+\s*;", header)) and not re.search(r"<=?\s*[A-Za-z_]", header)

    best = ONE
    max_depth = 0
    for idx, (s, e, header, body) in enumerate(loops):
        enclosing = [l for l in loops if l[0] < s and l[1] >= e]
        chain = enclosing + [loops[idx]]
        max_depth = max(max_depth, len(chain))
        t = ONE
        for (_, _, h, b) in chain:
            t = _mul(t, ONE if is_const(h) and not is_log(h, "") else (False, 0, 1) if is_log(h, b) else (False, 1, 0))
        best = _max(best, t)

    # sorting calls
    if re.search(r"\b(?:sort|stable_sort|qsort)\s*\(|\.sort\s*\(", masked):
        best = _max(best, (False, 1, 1))

    # recursion
    rec_calls, memo, rec_time, rec_space = 0, False, ONE, ONE
    for m in _FUNC_RX.finditer(masked):
        name = m.group(1)
        if name in _KEYWORDS:
            continue
        ob = m.end() - 1
        body = masked[ob:_match_brace(masked, ob) + 1]
        calls = re.findall(r"(?<![\w.>])%s\s*\(" % re.escape(name), body)
        if not calls:
            continue
        rec_calls += len(calls)
        has_memo = bool(re.search(r"\b(?:memo|cache|dp)\b|unordered_map|HashMap|containsKey|\.count\(", body))
        memo = memo or has_memo
        halving = bool(re.search(r"/\s*2|>>\s*1|\bmid\b", body))
        linear = any(re.search(r"\bfor\b|\bwhile\b", body) for _ in [0])
        if has_memo:
            t, s_ = (False, 1, 0), (False, 1, 0)
        elif len(calls) >= 2 and not halving:
            t, s_ = (True, 0, 0), (False, 1, 0)
        elif len(calls) >= 2:
            t, s_ = ((False, 1, 1) if linear else (False, 1, 0)), (False, 0, 1)
        elif halving:
            t, s_ = (False, 0, 1), (False, 0, 1)
        else:
            t, s_ = ((False, 2, 0) if linear else (False, 1, 0)), (False, 1, 0)
        rec_time, rec_space = _max(rec_time, t), _max(rec_space, s_)

    time_t = _max(best, rec_time)

    n_funcs = len([m for m in _FUNC_RX.finditer(masked) if m.group(1) not in _KEYWORDS])
    n_classes = len(re.findall(r"\b(?:class|struct|interface)\s+\w+", masked))
    conditions = len(re.findall(r"\b(?:if|switch|case)\b\s*[\(\w]", masked))

    # max nesting from braces
    depth = deepest = 0
    for ch in masked:
        if ch == "{":
            depth += 1
            deepest = max(deepest, depth)
        elif ch == "}":
            depth = max(0, depth - 1)

    structures = len(re.findall(r"\bnew\s+(?:ArrayList|HashMap|HashSet|LinkedList)|\bvector\s*<|\bunordered_(?:map|set)\s*<|\bmap\s*<|\bset\s*<|\[\s*\]\s*=|=\s*\[\]|=\s*\{\}", masked))
    grows = bool(re.search(r"\.(?:push|push_back|add|append|put|insert)\s*\(", masked)) and bool(loops)
    space_t = _max(rec_space, (False, 1, 0) if (grows or structures and loops) else ONE)
    if re.search(r"\bnew\s+\w+\s*\[[^\]]*\]\s*\[|vector\s*<\s*vector", masked):
        space_t = _max(space_t, (False, 2, 0))
    elif re.search(r"\bnew\s+\w+\s*\[\s*\w+\s*\]|\bvector\s*<[^>]+>\s+\w+\s*\(\s*\w+", masked):
        space_t = _max(space_t, (False, 1, 0))

    return {
        "time": time_t, "space": space_t, "recursive_calls": rec_calls, "memo": memo,
        "functions": n_funcs, "classes": n_classes, "loops": len(loops),
        "conditions": conditions, "depth": deepest,
        "structures": structures,
    }


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

_SPACE_NOTES = {
    (False, 0, 0): "O(1) — Constant (no growing data)",
    (False, 0, 1): "O(log n) — Logarithmic (call stack of a halving recursion)",
    (False, 1, 0): "O(n) — Linear (growing structure or recursion depth)",
    (False, 2, 0): "O(n²) — Quadratic (2D table)",
}


def _empty_result(lines, code_lines, comment_lines, time_text, space_text, **extra):
    base = {
        "total_lines": lines, "code_lines": code_lines, "comment_lines": comment_lines,
        "function_count": 0, "class_count": 0, "loop_count": 0, "condition_count": 0,
        "max_nesting": 0, "complexity_score": 0, "quality_score": 0,
        "time_complexity": time_text, "space_complexity": space_text,
        "recursive_calls": 0, "data_structures_used": 0,
    }
    base.update(extra)
    return base


def analyze_complexity(code, language="Python"):
    lang = normalize_language(language) or "python"
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    _, masked, comments = scan(code, lang)

    total_lines = len(code.split("\n"))
    code_lines = sum(1 for l in masked.split("\n") if l.strip())
    comment_lines = sum(1 for l in comments.split("\n") if l.strip())
    # a docstring is documentation too
    if lang == "python":
        comment_lines += len(re.findall(r'^\s*[rRuUbB]?(?:"""|\'\'\')', code, re.M)) // 1

    if lang == "sql":
        joins = len(re.findall(r"\bJOIN\b", masked, re.I))
        return _empty_result(
            total_lines, code_lines, comment_lines,
            "N/A — SQL cost depends on indexes and the query plan",
            "N/A — SQL memory use depends on the database engine",
            condition_count=len(re.findall(r"\bWHERE\b|\bCASE\b|\bHAVING\b", masked, re.I)),
            quality_score=max(0, 100 - (20 if comment_lines == 0 else 0)
                              - 10 * len(re.findall(r"SELECT\s+\*", masked, re.I))),
            data_structures_used=joins,
        )

    metrics = None
    if lang == "python":
        try:
            metrics = _python_metrics(code)
        except SyntaxError:
            metrics = None
    if metrics is None:
        metrics = _brace_metrics(masked, lang) if lang != "python" else _python_fallback(masked)

    time_t, space_t = metrics["time"], metrics["space"]
    time_text = _fmt(time_t)
    if metrics["recursive_calls"]:
        time_text += " (memoized recursion)" if metrics["memo"] else " (recursion)"
    space_text = _SPACE_NOTES.get(space_t, _fmt(space_t))

    # indentation based nesting for Python, brace based for the rest
    max_nesting = metrics["depth"]
    if lang == "python" and metrics["depth"] == 0:
        max_nesting = _py_indent_depth(masked)

    function_count, loops = metrics["functions"], metrics["loops"]
    conditions = metrics["conditions"]
    complexity_score = loops * 3 + conditions * 2 + max_nesting * 4 + function_count

    quality = 100
    if comment_lines == 0:
        quality -= 20
    if function_count == 0 and code_lines > 10:
        quality -= 15
    if max_nesting > 4:
        quality -= 20
    if loops > 5:
        quality -= 10
    if metrics["recursive_calls"] and function_count == 0:
        quality -= 10
    if metrics["structures"] > 5:
        quality -= 5

    return {
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "function_count": function_count,
        "class_count": metrics["classes"],
        "loop_count": loops,
        "condition_count": conditions,
        "max_nesting": max_nesting,
        "complexity_score": complexity_score,
        "quality_score": max(0, quality),
        "time_complexity": time_text,
        "space_complexity": space_text,
        "recursive_calls": metrics["recursive_calls"],
        "data_structures_used": metrics["structures"],
    }


def _py_indent_depth(masked):
    """Deepest indentation level (4 spaces or one tab = one level)."""
    deepest = 0
    for line in masked.split("\n"):
        if line.strip():
            lead = line[:len(line) - len(line.lstrip())].replace("\t", "    ")
            deepest = max(deepest, len(lead) // 4)
    return deepest


def _python_fallback(masked):
    """Code that does not parse (syntax error): count loops from text only."""
    loops = len(re.findall(r"^\s*(?:for|while)\b", masked, re.M))
    return {
        "time": (False, min(loops, 3), 0) if loops else ONE, "space": ONE,
        "recursive_calls": 0, "memo": False,
        "functions": len(re.findall(r"^\s*def\s", masked, re.M)),
        "classes": len(re.findall(r"^\s*class\s", masked, re.M)),
        "loops": loops, "conditions": len(re.findall(r"^\s*(?:if|elif)\b", masked, re.M)),
        "depth": 0, "structures": 0,
    }
