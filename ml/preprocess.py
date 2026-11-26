"""Shared source-code preprocessing.

Every analysis module needs to know which characters are real code and which
are comments or string contents. `scan()` answers that once, for all of them.
"""

SUPPORTED_LANGUAGES = ("python", "javascript", "java", "cpp", "sql")

_ALIASES = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript", "node": "javascript",
    "java": "java",
    "c++": "cpp", "cpp": "cpp", "c": "cpp", "cplusplus": "cpp",
    "sql": "sql",
}

# Names shown to users (the UI sends these).
DISPLAY_NAMES = {
    "python": "Python", "javascript": "JavaScript", "java": "Java",
    "cpp": "C++", "sql": "SQL",
}


def normalize_language(language):
    """Return one of SUPPORTED_LANGUAGES, or None if the name is not supported."""
    if not isinstance(language, str):
        return None
    return _ALIASES.get(language.strip().lower())


def _blank(ch):
    return ch if ch == "\n" else " "


def scan(code, lang):
    """Split source into three same-length views (newlines are always kept,
    so line numbers match the original text):

    code_view     comments removed, string contents kept
    masked_view   comments removed, string contents blanked ("" stays "")
    comment_view  only comments kept

    Matching rules against `masked_view` means words inside strings and
    comments can no longer trigger a rule.
    """
    n = len(code)
    cv, mv, cm = [], [], []

    def emit(c, m, k):
        cv.append(c)
        mv.append(m)
        cm.append(k)

    line_comment = {"python": "#", "sql": "--"}.get(lang, "//")
    block = None if lang == "python" else ("/*", "*/")
    quotes = {"python": "'\"", "javascript": "'\"`", "sql": "'\""}.get(lang, "'\"")
    triple = lang == "python"
    sql = lang == "sql"

    i = 0
    while i < n:
        c = code[i]

        # --- comments ---
        if code.startswith(line_comment, i):
            j = code.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                emit(" ", " ", code[k])
            i = j
            continue
        if block and code.startswith(block[0], i):
            j = code.find(block[1], i + 2)
            j = n if j == -1 else j + 2
            for k in range(i, j):
                ch = code[k]
                emit(_blank(ch), _blank(ch), ch)
            i = j
            continue

        # --- strings ---
        if c in quotes:
            q = c
            if triple and code.startswith(q * 3, i):
                end = code.find(q * 3, i + 3)
                body_end = n if end == -1 else end
                for _ in range(3):
                    emit(q, q, " ")
                for k in range(i + 3, body_end):
                    ch = code[k]
                    emit(ch, _blank(ch), _blank(ch))
                if end != -1:
                    for _ in range(3):
                        emit(q, q, " ")
                    i = end + 3
                else:
                    i = n
                continue
            emit(q, q, " ")
            i += 1
            multiline = q == "`"
            while i < n:
                ch = code[i]
                if ch == "\\" and not sql and i + 1 < n:
                    nx = code[i + 1]
                    emit(ch, " ", " ")
                    emit(nx, _blank(nx), _blank(nx))
                    i += 2
                    continue
                if ch == q:
                    if sql and i + 1 < n and code[i + 1] == q:  # '' escape
                        emit(ch, " ", " ")
                        emit(ch, " ", " ")
                        i += 2
                        continue
                    emit(q, q, " ")
                    i += 1
                    break
                if ch == "\n" and not multiline:
                    break  # unterminated string: stop at end of line
                emit(ch, _blank(ch), _blank(ch))
                i += 1
            continue

        emit(c, c, _blank(c))
        i += 1

    return "".join(cv), "".join(mv), "".join(cm)
