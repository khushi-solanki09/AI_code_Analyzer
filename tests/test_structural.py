"""Behaviour that only a real parser can get right."""

import pytest

from ml.bug_detector import detect_bugs, rule_counts


def found(code, lang="Python"):
    return {(b["line"], b["rule_id"], b["engine"]) for b in detect_bugs(code, lang)}


def ids(code, lang="Python"):
    return {rid for _, rid, _ in found(code, lang)}


# ── Python (ast) ─────────────────────────────────────────────────────

def test_import_alias_is_resolved():
    assert 6 in ids("import pickle as p\nd = p.load(f)\n")
    assert 6 in ids("from pickle import load as ld\nd = ld(f)\n")


def test_aliased_dangerous_call_not_confused_with_other_objects():
    assert 6 not in ids("d = loader.load(f)\n")
    assert 1 not in ids("model.eval()\n")


def test_multiline_call_is_found_at_its_first_line():
    code = "import subprocess\nsubprocess.run(\n    cmd,\n    shell=True,\n)\n"
    assert (2, 7, "ast") in found(code)


def test_shell_false_is_fine():
    assert 7 not in ids("import subprocess\nsubprocess.run(cmd, shell=False)\n")


def test_break_in_inner_loop_does_not_stop_outer_infinite_loop():
    code = "while True:\n    for i in a:\n        if i:\n            break\n"
    assert 13 in ids(code)


def test_break_in_the_loop_itself_is_an_exit():
    assert 13 not in ids("while True:\n    x = f()\n    if x:\n        break\n")


def test_return_and_raise_count_as_exits():
    assert 13 not in ids("def f():\n    while True:\n        return 1\n")
    assert 13 not in ids("while True:\n    raise SystemExit\n")


def test_secret_passed_as_keyword_argument():
    assert 3 in ids('db = connect(host="h", password="hunter2")\n')


def test_secret_in_class_attribute_and_dict_subscript():
    assert 4 in ids('class C:\n    secret_key = "abc"\n')
    assert 3 not in ids('cfg["password"] = os.environ["PW"]\n')


def test_empty_and_env_secrets_are_not_flagged():
    assert 3 not in ids('password = ""\n')
    assert 3 not in ids('password = os.getenv("PW")\n')


def test_multiline_mutable_default():
    assert 68 in ids("def f(a,\n      items=[]):\n    return items\n")


def test_open_inside_with_is_not_flagged_even_with_two_items():
    assert 11 not in ids("with open('a') as f, open('b') as g:\n    use(f, g)\n")


def test_unused_variable_ignores_underscore_and_used_values():
    assert 71 not in ids("def f():\n    _tmp = 1\n    return 2\n")
    assert 71 not in ids("def f():\n    n = 0\n    n += 1\n    return n\n")


def test_unused_import_respects_dunder_all():
    assert 70 not in ids("import os\n__all__ = ['os']\n")


def test_unreachable_only_flags_the_dead_statement():
    code = "def f():\n    return 1\n    a = 1\n    b = 2\n"
    assert {l for l, r, _ in found(code) if r == 69} == {3}


def test_syntax_error_is_reported_with_its_line():
    result = detect_bugs("x = 1\ndef f(:\n    pass\n", "Python")
    err = [b for b in result if b["rule_id"] == 74]
    assert err and err[0]["line"] == 2 and "Syntax error" in err[0]["message"]


def test_regex_fallback_still_finds_problems_in_broken_code():
    result = {(b["rule_id"], b["engine"]) for b in detect_bugs("def f(:\n  x = eval(s)\n", "Python")}
    assert (1, "regex") in result


def test_null_byte_and_pathological_input_do_not_crash():
    assert detect_bugs("x = 1\x00\n", "Python")
    assert isinstance(detect_bugs("x = " + "(" * 400 + "1" + ")" * 400, "Python"), list)
    assert isinstance(detect_bugs("x = " + " + ".join(["a"] * 3000) + "\n", "Python"), list)


def test_engine_is_reported():
    assert (1, 1, "ast") in found("y = eval(s)\n")


# ── JavaScript / Java / C++ (tree-sitter) ────────────────────────────

def test_js_words_in_strings_and_comments_do_not_match():
    code = 'const msg = "use console.log( and eval( here";\n// el.innerHTML = x\n'
    assert not ({1, 54, 55} & ids(code, "JavaScript"))


def test_js_findings_come_from_tree_sitter():
    assert (1, 1, "tree-sitter") in found("eval(x);\n", "JavaScript")


def test_js_hardcoded_secret_forms():
    assert 3 in ids('const password = "abc";\n', "JavaScript")
    assert 3 in ids('const o = { password: "abc" };\n', "JavaScript")
    assert 3 in ids('config.password = "abc";\n', "JavaScript")
    assert 3 not in ids('const password = process.env.PW;\n', "JavaScript")


def test_js_template_literal_sql_injection():
    assert 67 in ids("const q = `SELECT * FROM t WHERE id = ${id}`;\n", "JavaScript")
    assert 67 not in ids("const q = `SELECT * FROM t`;\n", "JavaScript")


def test_js_empty_catch_with_only_a_comment_still_counts():
    assert 51 in ids("try { f(); } catch (e) { /* ignore */ }\n", "JavaScript")
    assert 51 not in ids("try { f(); } catch (e) { log(e); }\n", "JavaScript")


def test_js_syntax_error_falls_back_to_regex():
    assert (1, 1, "regex") in found("if (x == ) { eval(a)\n", "JavaScript")


def test_java_catch_forms():
    assert 59 in ids("class A { void f() { try { g(); } catch (Exception e) { h(e); } } }", "Java")
    assert 59 not in ids("class A { void f() { try { g(); } catch (IOException e) { h(e); } } }", "Java")


def test_java_system_out_and_equals():
    assert 57 in ids('class A { void f() { System.out.println("x"); } }', "Java")
    assert 78 not in ids('class A { boolean f(String s) { return s.equals("a"); } }', "Java")


def test_java_sql_concat():
    code = 'class A { String q(String id) { return "SELECT * FROM t WHERE id = " + id; } }'
    assert 67 in ids(code, "Java")


def test_cpp_qualified_unsafe_calls():
    assert 61 in ids("void f() { std::strcpy(a, b); }", "C++")
    assert 61 not in ids("void f() { std::string s; }", "C++")


def test_cpp_using_namespace_and_new_array():
    assert 60 in ids("using namespace std;\n", "C++")
    assert 60 not in ids("using std::cout;\n", "C++")
    assert 63 in ids("void f(int n) { int* a = new int[n]; }", "C++")


def test_cpp_catch_all():
    assert 59 in ids("void f() { try { g(); } catch (...) { h(); } }", "C++")


# ── SQL (sqlparse) ───────────────────────────────────────────────────

def test_sql_select_star_vs_count_star():
    assert 64 in ids("SELECT * FROM users;", "SQL")
    assert 64 not in ids("SELECT COUNT(*) FROM users;", "SQL")


def test_sql_missing_where_reports_the_right_statement():
    code = "UPDATE a SET x = 1 WHERE id = 1;\nDELETE FROM b;\n"
    assert {l for l, r, _ in found(code, "SQL") if r == 66} == {2}


def test_sql_keywords_in_comments_and_strings_are_ignored():
    code = "-- DROP TABLE users;\nSELECT 'DROP TABLE x' AS note FROM t;\n"
    assert 65 not in ids(code, "SQL")


def test_sql_engine_is_sqlparse():
    assert (1, 65, "sqlparse") in found("DROP TABLE users;", "SQL")


# ── overall ──────────────────────────────────────────────────────────

def test_rule_counts_shape():
    counts = rule_counts()
    assert counts["total"] == 79
    assert counts["structural"]["python"] >= 55
    assert all(counts["structural"][l] >= 3 for l in ("python", "javascript", "java", "cpp", "sql"))


@pytest.mark.parametrize("lang", ["Python", "JavaScript", "Java", "C++", "SQL"])
def test_empty_and_whitespace_code_gives_no_findings(lang):
    assert detect_bugs("", lang) == [] and detect_bugs("   \n\n", lang) == []


def test_deeply_nested_code_degrades_instead_of_crashing():
    chain = "x = a" + ".b" * 2500 + "()\n"
    assert isinstance(detect_bugs(chain, "Python"), list)
    from ml.complexity import analyze_complexity
    assert "time_complexity" in analyze_complexity("x = " + " + ".join(["a"] * 3000) + "\n")
    assert "time_complexity" in analyze_complexity(chain)
