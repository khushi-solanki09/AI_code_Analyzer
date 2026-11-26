import json
import os

from ml.classifier import classify_code, model, training_data


def test_classifier_uses_equal_class_priors():
    # fit_prior=True would copy the 58/42 split of the training data
    nb = model.named_steps["classifier"]
    assert nb.fit_prior is False
    assert abs(nb.class_log_prior_[0] - nb.class_log_prior_[1]) < 1e-9


def test_training_set_size_matches_docs():
    X, y = training_data()
    assert len(X) == 151 and sum(y) == 88


def test_obviously_bad_and_good_code():
    assert classify_code('password = "admin123"\nexec(input())\n')["quality"] == "Needs Improvement"
    good = 'def area(r):\n    if r < 0:\n        raise ValueError("negative")\n    return 3.14159 * r * r\n'
    assert classify_code(good)["quality"] == "Good Code"


def test_output_shape():
    r = classify_code("x = 1")
    assert set(r) == {"quality", "confidence", "score"} and 0 <= r["score"] <= 100


def test_metrics_file_matches_training_set():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "metrics.json")
    with open(path) as f:
        m = json.load(f)
    assert m["samples"] == 151
    assert m["balanced_equal_priors"]["fit_prior"] is False
