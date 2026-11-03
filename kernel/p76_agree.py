#!/usr/bin/env python3
"""Answer-position fidelity to bf16, at n=1000, from the result files already on disk.

kernel/p72_fidelity.py found that the equalizer and the weight global scale both make the
model agree with bf16 MORE often while scoring LESS on MMLU-Pro. Over prompt tokens that
result is tight (11,958 positions). At the answer position it was measured on 64 questions,
which is +/-6 points, so it did not resolve -- and the answer position is the only one the
bench reads, so it is the only one that can settle anything.

No new GPU work is needed. `eval_bench` already writes `option_scores` (the debiased
per-option log-probs it takes its argmax from) and `pred_idx` for every question, so for any
two cells over the same question set the agreement is a join on `id`. That also removes the
split mismatch in p72, which sampled the validation split while the cells score the test
split: this compares the exact questions the accuracy numbers came from.

The arithmetic that makes this decisive: if a config picks bf16's option on a fraction `a`
of questions, and bf16 is right with probability p, then the config's accuracy is at least
`a * p` and rises with `a` unless its disagreements are systematically better than bf16 --
which for a quantized approximation of bf16 they cannot be. So accuracy must increase with
agreement. Any pair where agreement rises and accuracy falls is a pair where one of the two
numbers is not measuring what it claims.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys


def load(path: str):
    by_id = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:                                   # noqa: BLE001
            continue
        if "id" in d and "pred_idx" in d:
            by_id[d["id"]] = d
    return by_id


def kl_from_scores(p_scores, q_scores):
    """KL(bf16 || q) over options, from the averaged per-option log-probs.

    `option_scores` is a mean of log-probs over rotations, so it is not normalised; both
    sides are renormalised over the options before comparing, which is the distribution the
    argmax is taken from anyway.
    """
    def norm(v):
        m = max(v)
        e = [math.exp(x - m) for x in v]
        s = sum(e)
        return [x / s for x in e]
    p, q = norm(p_scores), norm(q_scores)
    return sum(pi * math.log(max(pi, 1e-12) / max(qi, 1e-12)) for pi, qi in zip(p, q))


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    emit = ""
    if "--emit" in argv:
        i = argv.index("--emit")
        emit = argv[i + 1]
        del argv[i:i + 2]
    d = argv[0] if len(argv) > 0 else "kernel/e2e_results_s28"
    ref_tag = argv[1] if len(argv) > 1 else "bf16_base"
    files = sorted(glob.glob(os.path.join(d, "*mmlu_pro*.jsonl")))
    cells = {}
    for f in files:
        tag = os.path.basename(f).replace(".jsonl", "").split("__")[-1]
        r = load(f)
        if r:
            cells[tag] = r
    if ref_tag not in cells:
        print(f"  no reference cell {ref_tag} in {d}")
        return 1
    ref = cells[ref_tag]
    print(f"  reference {ref_tag}: {len(ref)} questions\n")
    print(f"  {'tag':26s}{'acc':>7s}{'agree w/ bf16':>15s}{'KL(bf16||q)':>13s}"
          f"{'n':>7s}{'shared':>8s}")

    rows = []
    for tag, cell in cells.items():
        ids = [i for i in cell if i in ref]
        if len(ids) < 100:
            continue
        acc = 100.0 * sum(cell[i]["correct"] for i in ids) / len(ids)
        agree = 100.0 * sum(cell[i]["pred_idx"] == ref[i]["pred_idx"]
                            for i in ids) / len(ids)
        kls = [kl_from_scores(ref[i]["option_scores"], cell[i]["option_scores"])
               for i in ids if len(ref[i]["option_scores"]) ==
               len(cell[i]["option_scores"])]
        kl = sum(kls) / max(len(kls), 1)
        rows.append((agree, acc, kl, tag, len(cell), len(ids)))

    for agree, acc, kl, tag, n, sh in sorted(rows, key=lambda r: -r[0]):
        print(f"  {tag:26s}{acc:7.2f}{agree:15.2f}{kl:13.5f}{n:7d}{sh:8d}")

    print("\n  The prediction: accuracy rises with agreement, because agreeing with a"
          "\n  56.50-scoring reference is how a quantized copy of it scores well. Rows"
          "\n  where agreement is high and accuracy is low are rows where one of the two"
          "\n  numbers is not measuring what it claims.")
    ok = [r for r in rows if r[3] != ref_tag]
    if len(ok) >= 3:
        n = len(ok)
        ma = sum(r[0] for r in ok) / n
        mc = sum(r[1] for r in ok) / n
        cov = sum((r[0] - ma) * (r[1] - mc) for r in ok) / n
        sa = math.sqrt(sum((r[0] - ma) ** 2 for r in ok) / n)
        sc = math.sqrt(sum((r[1] - mc) ** 2 for r in ok) / n)
        print(f"\n  correlation(agreement, accuracy) over {n} cells = "
              f"{cov / max(sa * sc, 1e-12):+.3f}")
        print("  positive means the eval is consistent with the models' behaviour at the")
        print("  answer position; negative means it is not, and the deck is built on it.")

    if emit:
        # The figure (kernel/p78_tradeoff_mem.py) renders from this rather than from the
        # records, so it can be rebuilt on a box that has never seen the node.
        with open(emit, "w", encoding="utf-8") as fh:
            json.dump({"results_dir": d, "reference": ref_tag, "n_reference": len(ref),
                       "cells": {r[3]: {"accuracy": r[1], "agreement": r[0],
                                        "kl_from_bf16": r[2], "n": r[4], "shared": r[5]}
                                 for r in rows}}, fh, indent=2)
        print(f"\n  wrote {emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
