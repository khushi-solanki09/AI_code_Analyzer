import pytest

from ml.bug_detector import detect_bugs, get_bug_summary, rule_counts, rules_for


def hits(code, language="Python"):
    return sorted({(b["line"], b["rule_id"]) for b in detect_bugs(code, language)})


def rule_ids(code, language="Python"):
    return {rid for _, rid in hits(code, language)}


# (name, language, code, rule that must fire, line)
SHOULD_FIRE = [
    ("eval py", "Python", "y = eval(s)\n", 1, 1),
    ("eval js", "JavaScript", "var y = eval(s);\n", 1, 1),
    ("exec", "Python", "exec(code)\n", 2, 1),
    ("hardcoded password", "Python", 'password = "admin123"\n', 3, 1),
    ("hardcoded password dict key", "Python", 'cfg = {"password": "hunter2"}\n', 3, 1),
    ("hardcoded password c++", "C++", 'char password[] = "admin123";\n', 3, 1),
    ("hardcoded secret", "JavaScript", 'const secret = "abc123";\n', 4, 1),
    ("hardcoded api key", "JavaScript", 'var apiKey = "sk-abc123xyz";\n', 5, 1),
    ("pickle", "Python", "d = pickle.load(f)\n", 6, 1),
    ("shell true", "Python", "subprocess.run(c, shell=True)\n", 7, 1),
    ("__import__", "Python", "m = __import__(name)\n", 8, 1),
    ("bare except", "Python", "try:\n    f()\nexcept:\n    log()\n", 9, 3),
    ("except Exception", "Python", "try:\n    f()\nexcept Exception:\n    log()\n", 10, 3),
    ("open without with", "Python", 'f = open("a.txt")\n', 11, 1),
    ("global", "Python", "def f():\n    global x\n", 12, 2),
    ("while true no exit", "Python", "while True:\n    work()\n", 13, 1),
    ("sleep", "Python", "time.sleep(1)\n", 14, 1),
    ("assert", "Python", "assert x > 0\n", 15, 1),
    ("wildcard import", "Python", "from os import *\n", 16, 1),
    ("wildcard import java", "Java", "import java.util.*;\n", 16, 1),
    ("raw input", "Python", 'name = input("who? ")\n', 17, 1),
    ("division by zero", "Python", "x = 5 / 0\n", 18, 1),
    ("division by zero java", "Java", "int x = a / 0;\n", 18, 1),
    ("int(input())", "Python", "n = int(input())\n", 19, 1),
    ("float(input())", "Python", "n = float(input())\n", 20, 1),
    ("== None", "Python", "if x == None:\n    pass\n", 21, 1),
    ("== True", "Python", "if x == True:\n    pass\n", 22, 1),
    ("== False", "Python", "if x == False:\n    pass\n", 23, 1),
    ("print", "Python", 'print("hi")\n', 24, 1),
    ("todo in comment", "Python", "x = 1  # TODO fix this\n", 25, 1),
    ("todo in js comment", "JavaScript", "// FIXME later\nlet a = 1;\n", 25, 1),
    ("pass block", "Python", "def f():\n    pass\n", 26, 2),
    ("range(len())", "Python", "for i in range(len(a)):\n    use(i)\n", 27, 1),
    ("x = x + 1", "Python", "count = count + 1\n", 28, 1),
    ("if len()", "Python", "if len(items) > 0:\n    go()\n", 29, 1),
    ("type()", "Python", "if type(x) == int:\n    go()\n", 30, 1),
    ("+ str()", "Python", 'msg = "n=" + str(n)\n', 31, 1),
    ("append in loop", "Python", "r = []\nfor i in a:\n    r.append(i)\n", 32, 3),
    ("not in range", "Python", "if x not in range(10):\n    go()\n", 33, 1),
    ("sorted(sorted())", "Python", "y = sorted(sorted(a))\n", 34, 1),
    ("dict()", "Python", "d = dict()\n", 35, 1),
    ("def function", "Python", "def function(x):\n    return x\n", 36, 1),
    ("lst[len(lst)]", "Python", "v = lst[len(lst)]\n", 37, 1),
    ("arr[arr.length]", "JavaScript", "var v = arr[arr.length];\n", 37, 1),
    ("self.self", "Python", "self.self.x = 1\n", 38, 1),
    ("recursion without negative guard", "Python",
     "def fact(n):\n    if n == 0:\n        return 1\n    return n * fact(n - 1)\n", 39, 2),
    ("shadow sum", "Python", "sum = 0\n", 40, 1),
    ("shadow dict", "Python", "dict = {}\n", 41, 1),
    ("shadow str", "Python", 'str = "x"\n', 42, 1),
    ("shadow list", "Python", "list = [1]\n", 43, 1),
    ("init self self", "Python", "class A:\n    def __init__(self, self):\n        pass\n", 44, 2),
    ("elif True", "Python", "if a:\n    x()\nelif True:\n    y()\n", 45, 3),
    ("assignment in if", "Python", "if x = 5:\n    go()\n", 46, 1),
    ("assignment in if js", "JavaScript", "if (x = 5) { go(); }\n", 46, 1),
    ("except continue", "Python", "for i in a:\n    try:\n        f()\n    except ValueError:\n        continue\n", 47, 4),
    ("import a, b", "Python", "import os, sys\n", 48, 1),
    ("return True/False", "Python", "def f(x):\n    if x:\n        return True\n    return False\n", 49, 3),
    ("except pass", "Python", "try:\n    f()\nexcept ValueError:\n    pass\n", 50, 3),
    ("empty catch java", "Java", "try { f(); } catch (IOException e) { }\n", 51, 1),
    ("var", "JavaScript", "var x = 1;\n", 52, 1),
    ("loose equality", "JavaScript", "if (a == b) {}\n", 53, 1),
    ("console.log", "JavaScript", 'console.log("x");\n', 54, 1),
    ("innerHTML", "JavaScript", "el.innerHTML = userText;\n", 55, 1),
    ("document.write", "JavaScript", 'document.write("x");\n', 56, 1),
    ("System.out", "Java", 'System.out.println("x");\n', 57, 1),
    ("printStackTrace", "Java", "e.printStackTrace();\n", 58, 1),
    ("catch Exception java", "Java", "try { f(); } catch (Exception e) { log(e); }\n", 59, 1),
    ("using namespace std", "C++", "using namespace std;\n", 60, 1),
    ("strcpy", "C++", "strcpy(dst, src);\n", 61, 1),
    ("os.system", "Python", 'os.system("ls")\n', 62, 1),
    ("system() c++", "C++", "system(cmd);\n", 62, 1),
    ("malloc", "C++", "int* p = (int*)malloc(10);\n", 63, 1),
    ("new[]", "C++", "int* a = new int[n];\n", 63, 1),
    ("select star", "SQL", "SELECT * FROM users;\n", 64, 1),
    ("drop table", "SQL", "DROP TABLE users;\n", 65, 1),
    ("delete without where", "SQL", "DELETE FROM users;\n", 66, 1),
    ("update without where", "SQL", "UPDATE users SET a = 1;\n", 66, 1),
    ("sql injection py concat", "Python", 'q = "SELECT * FROM users WHERE id = " + uid\n', 67, 1),
    ("sql injection py fstring", "Python", 'q = f"SELECT * FROM users WHERE id = {uid}"\n', 67, 1),
    ("sql injection js template", "JavaScript", "const q = `SELECT * FROM users WHERE id = ${id}`;\n", 67, 1),
    ("mutable default", "Python", "def f(x, items=[]):\n    return items\n", 68, 1),
    ("unreachable", "Python", "def f(x):\n    return x\n    y = 1\n", 69, 3),
]


@pytest.mark.parametrize("name,lang,code,rid,line", SHOULD_FIRE, ids=[c[0] for c in SHOULD_FIRE])
def test_rule_fires(name, lang, code, rid, line):
    assert (line, rid) in hits(code, lang)


# (name, language, code, rule that must NOT fire)
SHOULD_NOT_FIRE = [
    ("env password is fine", "Python", 'password = os.environ.get("PASSWORD")\n', 3),
    ("empty password string", "Python", 'password = ""\n', 3),
    ("word password in a string", "Python", 'print("password: ", name)\n', 3),
    ("eval in comment", "Python", "# never call eval(x)\nx = 1\n", 1),
    ("eval in string", "Python", 's = "eval(x)"\n', 1),
    ("literal_eval is safe", "Python", "x = ast.literal_eval(s)\n", 1),
    ("method named eval", "Python", "model.eval()\n", 1),
    ("python rule on java code", "Java", "int x = eval(1);\n", 1),
    ("python rule on sql", "SQL", "SELECT 1 /* eval( */;\n", 1),
    ("with open is fine", "Python", 'with open("a") as f:\n    data = f.read()\n', 11),
    ("while True with break", "Python", "while True:\n    x = f()\n    if x:\n        break\n", 13),
    ("urlopen is not open", "Python", "r = urlopen(u)\n", 11),
    ("append outside loop", "Python", "r = []\nr.append(1)\n", 32),
    ("fib with <= guard", "Python",
     "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n", 39),
    ("factorial with negative check", "Python",
     "def fact(n):\n    if n < 0:\n        raise ValueError\n    if n == 0:\n        return 1\n    return n * fact(n - 1)\n", 39),
    ("TODO inside string", "Python", 'x = "TODO"\n', 25),
    ("=== is fine in js", "JavaScript", "if (a === b) {}\n", 53),
    ("!== is fine in js", "JavaScript", "if (a !== b) {}\n", 53),
    ("division by 0.5", "Python", "y = x / 0.5\n", 18),
    ("shadowed name is not a call", "Python", "my_list = list(range(3))\n", 43),
    ("int(input()) not double reported as input()", "Python", "n = int(input())\n", 17),
    ("sql with where", "SQL", "UPDATE t SET a = 1 WHERE id = 2;\n", 66),
    ("parameterised query", "Python", 'cur.execute("SELECT * FROM t WHERE id = %s", (uid,))\n', 67),
    ("plain update sentence", "Python", 'msg = "Update complete: " + str(n)\n', 67),
    ("code after if-return", "Python", "def f(x):\n    if x:\n        return 1\n    return 2\n", 69),
    ("multi-line return", "Python", "def f():\n    return [\n        1,\n    ]\n", 69),
    ("default None is fine", "Python", "def f(x, items=None):\n    return items\n", 68),
]


@pytest.mark.parametrize("name,lang,code,rid", SHOULD_NOT_FIRE, ids=[c[0] for c in SHOULD_NOT_FIRE])
def test_rule_stays_quiet(name, lang, code, rid):
    assert rid not in rule_ids(code, lang)


def test_every_rule_has_a_positive_test():
    tested = {c[3] for c in SHOULD_FIRE}
    assert tested == {r["id"] for r in __import__("ml.bug_detector", fromlist=["RULES"]).RULES}


def test_rule_ids_are_unique():
    from ml.bug_detector import RULES
    ids = [r["id"] for r in RULES]
    assert len(ids) == len(set(ids))


def test_every_language_has_rules():
    counts = rule_counts()
    for lang in ("python", "javascript", "java", "cpp", "sql"):
        assert counts[lang] >= 8


def test_rule_50_is_not_a_dead_pattern():
    # the old rule 50 searched for the literal text "bare_except"
    assert 50 in rule_ids("try:\n    f()\nexcept ValueError:\n    pass\n")


def test_language_aliases():
    assert detect_bugs("x = eval(s)", "py") == detect_bugs("x = eval(s)", "Python")
    assert rules_for("C++") == rules_for("cpp")


def test_unsupported_language_raises():
    with pytest.raises(ValueError):
        detect_bugs("x", "Rust")


def test_windows_line_endings():
    assert (2, 1) in hits("a = 1\r\nb = eval(x)\r\n")


def test_line_numbers_survive_multiline_strings():
    code = 'doc = """\nline\neval(x)\n"""\ny = eval(z)\n'
    assert hits(code) == [(5, 1)]


def test_finding_shape_and_summary():
    bugs = detect_bugs('password = "x"\nx = 1 / 0\n')
    assert {"rule_id", "line", "code", "severity", "message", "category", "fix"} <= set(bugs[0])
    s = get_bug_summary(bugs)
    assert s["total"] == len(bugs) == s["high"] + s["medium"] + s["low"]
