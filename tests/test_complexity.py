import pytest

from ml.complexity import analyze_complexity


def t(code, lang="Python"):
    return analyze_complexity(code, lang)["time_complexity"]


def s(code, lang="Python"):
    return analyze_complexity(code, lang)["space_complexity"]


CASES = [
    ("constant", "Python", "x = 1\ny = 2\n", "O(1)"),
    ("linear loop", "Python", "def f(a):\n    for x in a:\n        use(x)\n", "O(n)"),
    ("nested loops", "Python", "for x in a:\n    for y in a:\n        pass\n", "O(n²)"),
    ("triple loops", "Python", "for i in a:\n  for j in a:\n    for k in a:\n      pass\n", "O(n³)"),
    ("constant range loops", "Python", "for i in range(10):\n    for j in range(10):\n        pass\n", "O(1)"),
    ("halving while loop", "Python", "def f(n):\n    while n > 1:\n        n //= 2\n", "O(log n)"),
    ("doubling while loop", "Python", "i = 1\nwhile i < n:\n    i *= 2\n", "O(log n)"),
    ("binary search loop", "Python",
     "def bs(a, t):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
     "        if a[mid] == t:\n            return mid\n        elif a[mid] < t:\n            lo = mid + 1\n"
     "        else:\n            hi = mid - 1\n", "O(log n)"),
    ("naive fibonacci", "Python",
     "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n", "O(2^n)"),
    ("fibonacci with lru_cache", "Python",
     "from functools import lru_cache\n@lru_cache(None)\ndef fib(n):\n    if n <= 1:\n        return n\n"
     "    return fib(n - 1) + fib(n - 2)\n", "O(n)"),
    ("fibonacci with memo dict", "Python",
     "def fib(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n"
     "    memo[n] = fib(n - 1) + fib(n - 2)\n    return memo[n]\n", "O(n)"),
    ("factorial recursion", "Python", "def f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)\n", "O(n)"),
    ("merge sort", "Python",
     "def ms(a):\n    if len(a) <= 1:\n        return a\n    mid = len(a) // 2\n    l = ms(a[:mid])\n    r = ms(a[mid:])\n"
     "    out = []\n    i = j = 0\n    while i < len(l) and j < len(r):\n        if l[i] < r[j]:\n"
     "            out.append(l[i])\n            i += 1\n        else:\n            out.append(r[j])\n"
     "            j += 1\n    return out + l[i:] + r[j:]\n", "O(n log n)"),
    ("slicing recursion copies", "Python", "def s(l):\n    if not l:\n        return 0\n    return l[0] + s(l[1:])\n", "O(n²)"),
    ("sorted builtin", "Python", "x = sorted(data)\n", "O(n log n)"),
    ("linear builtin inside loop", "Python", "for i in a:\n    t = sum(a)\n", "O(n²)"),
    ("comment mentioning loops", "Python", "# for i in range(n): for j in range(n)\nx = 1\n", "O(1)"),
    ("js nested", "JavaScript",
     "function f(a){\n for (let i=0;i<a.length;i++){\n  for (let j=0;j<a.length;j++){ x++; }\n }\n}\n", "O(n²)"),
    ("js doubling", "JavaScript", "function f(n){ let c=0; for (let i=1;i<n;i*=2){ c++; } return c; }", "O(log n)"),
    ("js naive fib", "JavaScript", "function fib(n){ if(n<=1) return n; return fib(n-1)+fib(n-2); }", "O(2^n)"),
    ("js memoized fib", "JavaScript",
     "const memo = {};\nfunction fib(n){ if(n in memo) return memo[n]; if(n<=1) return n; memo[n]=fib(n-1)+fib(n-2); return memo[n]; }",
     "O(n)"),
    ("java triple", "Java",
     "public void f(int n){\n for(int i=0;i<n;i++){ for(int j=0;j<n;j++){ for(int k=0;k<n;k++){ s++; } } }\n}", "O(n³)"),
    ("c++ constant loops", "C++", "int main(){ for(int i=0;i<10;i++){ for(int j=0;j<10;j++){ x++; } } return 0; }", "O(1)"),
    ("c++ sort", "C++", "void f(vector<int>& v){ sort(v.begin(), v.end()); }", "O(n log n)"),
]


@pytest.mark.parametrize("name,lang,code,expected", CASES, ids=[c[0] for c in CASES])
def test_time_complexity(name, lang, code, expected):
    assert t(code, lang).startswith(expected), t(code, lang)


def test_space_2d_table():
    assert s("dp = [[0] * n for _ in range(n)]\n").startswith("O(n²)")


def test_space_constant():
    assert s("x = 1\n").startswith("O(1)")


def test_sql_is_not_given_a_made_up_big_o():
    assert t("SELECT * FROM a;", "SQL").startswith("N/A")


def test_syntax_error_does_not_crash():
    r = analyze_complexity("def f(:\n  for x in y:\n   pass")
    assert r["loop_count"] == 1


def test_loops_inside_comments_and_strings_are_not_counted():
    r = analyze_complexity('# while True:\ns = "for i in range(3)"\n')
    assert r["loop_count"] == 0


def test_metrics_shape():
    r = analyze_complexity("def f(a):\n    # note\n    for x in a:\n        if x:\n            pass\n")
    assert r["function_count"] == 1 and r["loop_count"] == 1 and r["condition_count"] == 1
    assert r["comment_lines"] == 1 and r["max_nesting"] == 3
