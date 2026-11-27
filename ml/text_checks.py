"""Line-based helpers used when a file cannot be parsed (the regex fallback)
and by rules that only need text. They work on the "masked" view of the
source, where comments are removed and string contents are blanked.
"""

import re


def indent(line):
    return len(line) - len(line.lstrip())


def in_loop(lines, i):
    """True if line i sits (at any depth) inside a Python for/while loop."""
    ind = indent(lines[i])
    for j in range(i - 1, -1, -1):
        s = lines[j]
        if not s.strip():
            continue
        if indent(s) < ind:
            if re.match(r"\s*(?:async\s+)?(?:for|while)\b", s):
                return True
            ind = indent(s)
            if ind == 0:
                break
    return False


_EXIT_RX = re.compile(r"\b(?:break|return|raise)\b|\bsys\.exit\b|\bexit\s*\(|\bos\._exit\b")


def loop_has_exit(lines, i):
    """True if the Python loop that starts on line i contains break/return/raise."""
    base = indent(lines[i])
    for j in range(i + 1, len(lines)):
        s = lines[j]
        if not s.strip():
            continue
        if indent(s) <= base:
            break
        if _EXIT_RX.search(s):
            return True
    return False


_DEF_RX = re.compile(r"^(\s*)def\s+(\w+)\s*\(([^)]*)\)")


def missing_negative_guard(lines):
    """Recursive function whose only base case is `n == 0`."""
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
            if s.strip() and indent(s) <= base:
                break
            body.append((j, s))
            j += 1
        text = "\n".join(s for _, s in body)
        if not re.search(r"(?<![\w.])%s\s*\(" % re.escape(name), text):
            continue
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


def unreachable(lines):
    """Statement directly after return/raise/break/continue in the same block."""
    hits = []
    for i, line in enumerate(lines):
        if not re.match(r"^\s*(?:return|raise|break|continue)\b", line):
            continue
        if line.rstrip().endswith(":"):
            continue
        if any(line.count(a) > line.count(b) for a, b in ("()", "[]", "{}")):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            continue
        nxt = lines[j]
        if (indent(nxt) == indent(line)
                and not re.match(r"\s*(?:else|elif|except|finally|case)\b", nxt)
                and not re.match(r"\s*[)\]}]", nxt)):
            hits.append(j)
    return hits


TEXT_CHECKS = {
    "missing_negative_guard": missing_negative_guard,
    "unreachable": unreachable,
}
