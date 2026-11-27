import copy

import pytest
import yaml

from ml.rule_loader import RULES_PATH, RuleConfigError, load_rules


@pytest.fixture()
def raw():
    with open(RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write(tmp_path, data):
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_shipped_rules_load():
    rules = load_rules()
    assert len(rules) == 79 and len({r.id for r in rules}) == 79


def test_every_rule_has_message_fix_and_an_implementation():
    for r in load_rules():
        assert r.message.strip() and r.fix.strip() and r.impls


def _rule(data, rid):
    return next(r for r in data["rules"] if r["id"] == rid)


def test_duplicate_id_is_rejected(raw, tmp_path):
    bad = copy.deepcopy(raw)
    bad["rules"].append(copy.deepcopy(bad["rules"][0]))
    with pytest.raises(RuleConfigError, match="duplicate rule id"):
        load_rules(write(tmp_path, bad))


def test_bad_severity_is_rejected(raw, tmp_path):
    bad = copy.deepcopy(raw)
    _rule(bad, 1)["severity"] = "SEVERE"
    with pytest.raises(RuleConfigError, match="severity"):
        load_rules(write(tmp_path, bad))


def test_unknown_ast_check_is_rejected(raw, tmp_path):
    bad = copy.deepcopy(raw)
    _rule(bad, 1)["languages"]["python"]["ast"]["check"] = "does_not_exist"
    with pytest.raises(RuleConfigError, match="unknown ast check"):
        load_rules(write(tmp_path, bad))


def test_invalid_regex_is_rejected(raw, tmp_path):
    bad = copy.deepcopy(raw)
    _rule(bad, 1)["languages"]["python"]["regex"] = "(unclosed"
    with pytest.raises(RuleConfigError, match="invalid regex"):
        load_rules(write(tmp_path, bad))


def test_invalid_tree_sitter_query_is_rejected(raw, tmp_path):
    bad = copy.deepcopy(raw)
    _rule(bad, 1)["languages"]["javascript"]["query"] = "(no_such_node_type) @hit"
    with pytest.raises(RuleConfigError, match="invalid tree-sitter query"):
        load_rules(write(tmp_path, bad))


def test_unknown_language_is_rejected(raw, tmp_path):
    bad = copy.deepcopy(raw)
    _rule(bad, 1)["languages"]["rust"] = {"regex": "x"}
    with pytest.raises(RuleConfigError, match="unknown language"):
        load_rules(write(tmp_path, bad))


def test_ast_on_a_non_python_language_is_rejected(raw, tmp_path):
    bad = copy.deepcopy(raw)
    _rule(bad, 1)["languages"]["javascript"]["ast"] = {"check": "calls", "names": ["eval"]}
    with pytest.raises(RuleConfigError, match="only for python"):
        load_rules(write(tmp_path, bad))


def test_all_problems_are_listed_together(raw, tmp_path):
    bad = copy.deepcopy(raw)
    _rule(bad, 1)["severity"] = "NOPE"
    _rule(bad, 2)["fix"] = ""
    with pytest.raises(RuleConfigError) as err:
        load_rules(write(tmp_path, bad))
    assert "rule 1" in str(err.value) and "rule 2" in str(err.value)


def test_missing_rules_list_is_rejected(tmp_path):
    with pytest.raises(RuleConfigError):
        load_rules(write(tmp_path, {"version": 1}))
