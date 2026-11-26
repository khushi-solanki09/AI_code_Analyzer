"""Honest evaluation of the code-quality classifier.

Run:  python -m ml.evaluate

What it reports (and why):
* Repeated stratified 5-fold cross-validation (10 repeats). One 5-fold run on
  151 samples is noisy; repeating it gives a mean and a spread.
* A confusion matrix and per-class precision/recall. Overall accuracy hides a
  model that leans towards one class.
* A comparison of the old setup (class prior learned from the data) with the
  balanced one (equal priors).
* A near-duplicate check. Many samples are variations of the same snippet, so
  a held-out fold often contains a close twin of a training sample and the
  score is optimistic. The number of such pairs is printed so nobody has to
  guess.

Results are written to ml/metrics.json.
"""

import json
import os
import re
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold

from ml.classifier import build_pipeline, training_data

warnings.filterwarnings("ignore")

CLASS_NAMES = {0: "Needs Improvement", 1: "Good Code"}


def _tokens(text):
    return set(re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", text))


def near_duplicate_pairs(X, threshold=0.8):
    toks = [_tokens(x) for x in X]
    pairs = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            union = len(toks[i] | toks[j])
            if union and len(toks[i] & toks[j]) / union >= threshold:
                pairs += 1
    return pairs


def evaluate(fit_prior, repeats=10, seed=42):
    X, y = training_data()
    X, y = np.array(X, dtype=object), np.array(y)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    per_run = []
    total_cm = np.zeros((2, 2), dtype=int)
    for train_idx, test_idx in rskf.split(X, y):
        model = build_pipeline(fit_prior)
        model.fit(list(X[train_idx]), y[train_idx])
        pred = model.predict(list(X[test_idx]))
        total_cm += confusion_matrix(y[test_idx], pred, labels=[0, 1])
        p, r, f, _ = precision_recall_fscore_support(y[test_idx], pred, labels=[0, 1], zero_division=0)
        per_run.append({"acc": float((pred == y[test_idx]).mean()),
                        "p0": p[0], "r0": r[0], "f0": f[0], "p1": p[1], "r1": r[1], "f1": f[1]})

    def stat(key):
        vals = np.array([r[key] for r in per_run]) * 100
        return {"mean": round(float(vals.mean()), 1), "std": round(float(vals.std()), 1)}

    return {
        "fit_prior": fit_prior,
        "accuracy": stat("acc"),
        "needs_improvement": {"precision": stat("p0"), "recall": stat("r0"), "f1": stat("f0")},
        "good_code": {"precision": stat("p1"), "recall": stat("r1"), "f1": stat("f1")},
        "confusion_matrix_total": {
            "rows_actual_cols_predicted": ["Needs Improvement", "Good Code"],
            "matrix": total_cm.tolist(),
        },
    }


def main():
    X, y = training_data()
    counts = {CLASS_NAMES[c]: int((np.array(y) == c).sum()) for c in (0, 1)}
    report = {
        "samples": len(X),
        "class_counts": counts,
        "near_duplicate_pairs_jaccard_0.8": near_duplicate_pairs(X),
        "method": "repeated stratified 5-fold cross-validation, 10 repeats, seed 42",
        "baseline_prior_from_data": evaluate(fit_prior=True),
        "balanced_equal_priors": evaluate(fit_prior=False),
        "caveats": [
            "Only %d hand-written samples, so the numbers are indicative, not proof of general quality." % len(X),
            "Good/bad labels were assigned by hand and reflect one person's idea of quality.",
            "All samples were written by one person in a similar style, so style leaks across folds. No near-duplicate pairs were found, but real-world code will look different.",
        ],
    }
    path = os.path.join(os.path.dirname(__file__), "metrics.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print("Samples:", len(X), counts)
    print("Near-duplicate pairs (token Jaccard >= 0.8):", report["near_duplicate_pairs_jaccard_0.8"])
    for name in ("baseline_prior_from_data", "balanced_equal_priors"):
        r = report[name]
        print("\n%s  (fit_prior=%s)" % (name, r["fit_prior"]))
        print("  accuracy            %5.1f +/- %.1f" % (r["accuracy"]["mean"], r["accuracy"]["std"]))
        for cls in ("needs_improvement", "good_code"):
            m = r[cls]
            print("  %-18s  precision %5.1f   recall %5.1f   f1 %5.1f" % (
                cls, m["precision"]["mean"], m["recall"]["mean"], m["f1"]["mean"]))
        print("  confusion matrix (rows=actual, cols=predicted [NI, Good]):",
              r["confusion_matrix_total"]["matrix"])
    print("\nSaved", path)


if __name__ == "__main__":
    main()
