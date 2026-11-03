#!/usr/bin/env python3
"""Paired analysis of the accuracy runs.

Every arm sees the same questions in the same order with deterministic scoring, so the
right comparison is paired, not two independent binomial intervals. With N=198 an
unpaired 95% interval is about +/-7 points and would call a real 3-point regression
"not significant"; McNemar on the per-question disagreements resolves the same effect
because it conditions on the questions where the two arms actually differ.

Reported per (benchmark, arm):
  * accuracy and the paired delta vs bf16
  * McNemar exact two-sided p on the discordant pairs (b = bf16 right / arm wrong,
    c = bf16 wrong / arm right)
  * a paired bootstrap 95% CI on the delta
  * mean margin between the chosen and the gold letter logprob, which moves before
    accuracy does and so gives an early warning of damage that has not yet flipped
    an answer
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

_BASE_ORDER = ["bf16", "fp4", "w4a4_tc", "w4a6", "w6a6_tc", "w6a6", "fp6"]
# _eq is the equalizer calibrated on prose, _eq2 the same equalizer calibrated in domain.
# Both are kept: the first one made fp6 worse end to end, and the pair beside the plain arm
# is what shows that the calibration data, not the equalizer, was the defect.
_EQ_TAGS = ["", "_eq", "_eq2"]
# Equalized arms sit beside their plain counterpart so the fold's effect is read down the
# column rather than by hunting for a second table.
ARM_ORDER = ["bf16"] + [b + t for b in _BASE_ORDER[1:] for t in _EQ_TAGS]
# Weight bits per arm, which is the axis the comparison is actually about.
_BITS = {"bf16": 16, "fp4": 4, "w4a4_tc": 4, "w4a6": 4, "w6a6_tc": 4, "w6a6": 6,
         "fp6": 6}


def base_arm(a: str) -> str:
    for t in ("_eq2", "_eq"):
        if a.endswith(t):
            return a[: -len(t)]
    return a


ARM_BITS = {a: _BITS[base_arm(a)] for a in ARM_ORDER}
ARM_NOTE = {
    "fp4": "MXFP4 w + MXFP4 a, E8M0 /32",
    "w4a4_tc": "NVFP4 w + NVFP4 a, E4M3 /32 (both transcoded)",
    "w4a6": "MXFP4 w + MXFP6 a, E8M0 /32",
    "w6a6_tc": "NVFP4 w + MXFP6 a, E4M3 /32 (w transcoded)  [user's w4a6_tc]",
    "w6a6": "MXFP6 w + MXFP6 a, E8M0 /32 -- 6-bit at the same granularity as the rest",
    "fp6": "MXFP6 w + MXFP6 a, per-token/channel f32 -- the only arm NOT block scaled",
}
EQ_NOTE = ("_eq/_eq2 = per-input-channel equalization (kernel/qsmooth.py): W*s folded into "
           "the stored\n        weight, 1/s applied in the activation quantizer. Exact in "
           "exact arithmetic, so it changes\n        what the quantizer sees and nothing "
           "else. _eq calibrated the activation statistics on prose;\n        _eq2 on prose "
           "plus MMLU-Pro validation prompts, disjoint from the scored items. The\n        "
           "difference between those two columns is the difference between an equalizer that "
           "amplifies\n        the channels it mispredicts and one that does not -- see the "
           "kernel/qsmooth.py docstring.")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs, computed in log space.

    The direct form -- sum(comb(n, i) for i in range(k+1)) / 2**n -- is correct and overflows:
    `2.0 ** n` is inf above n = 1024, and CPython raises rather than returning it. So this
    function worked on every sample size the project had used and failed on the first one large
    enough for the test to be interesting. s37 scores all 12,032 MMLU-Pro items, where the
    discordant counts run 279-2329, and `best_e2e report` crashed on the whole registry -- not
    on the one cell, on all of them, because a single raise takes down the pass.

    Both terms are evaluated as logs and recombined against the largest of them, which has no
    ceiling and costs O(k). Values agree with the old form to ~1e-15 wherever the old form ran.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    ln2 = math.log(2.0)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * ln2
            for i in range(k + 1)]
    mx = max(logs)
    return min(1.0, 2.0 * math.exp(mx + math.log(sum(math.exp(v - mx) for v in logs))))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down: which of a family of tests survive at alpha."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            out[i] = True
        else:
            break
    return out


def paired_bootstrap(x: np.ndarray, y: np.ndarray, iters=10000, seed=0):
    """95% CI of mean(y) - mean(x), resampling QUESTIONS (keeping pairs together)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(iters, n))
    d = (y[idx] - x[idx]).mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load(out_dir, repo):
    """(bench, arm) -> {qid: record}"""
    base = os.path.basename(repo)
    data = defaultdict(dict)
    for p in glob.glob(os.path.join(out_dir, f"{base}__*.jsonl")):
        stem = os.path.basename(p)[:-6]
        _, bench, arm = stem.split("__")
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                data[(bench, arm)][r["id"]] = r
    return data


def gold_margin(r):
    """score(chosen) - score(gold). 0 when correct; size = confidence in the wrong
    answer. Averaged, it is a continuous damage measure that moves before accuracy does.

    Reads `option_scores`/`gold_idx` when present (the debiased harness, where scores are
    indexed by option rather than by letter slot) and falls back to the letter-indexed
    fields so files written before the rewrite still analyse.
    """
    lp = r.get("option_scores") or r.get("letter_logprobs")
    if not lp:
        return None
    gi = r.get("gold_idx")
    if gi is None:
        letters = "ABCDEFGHIJ"
        gi = letters.index(r["gold"]) if r["gold"] in letters else None
    if gi is None or gi >= len(lp):
        return None
    return max(lp) - lp[gi]


def slot_bias(rows):
    """(fraction on the most-chosen slot, uniform fraction) over predicted answers.

    The measurement that started this: on GPQA the bf16 model put 67% of its answers on
    slot A against a 25% base rate, which is what a model does when it does not know and
    falls back to a position prior. Reported per arm because quantization moves that prior
    around, and an accuracy change driven by it is not a knowledge change.
    """
    from collections import Counter
    c = Counter(r.get("pred_idx", r.get("choice")) for r in rows)
    n = max(1, sum(c.values()))
    nopt = max((r.get("n_options") or len(r.get("option_scores") or
                                          r.get("letter_logprobs") or [1]))
               for r in rows)
    return max(c.values()) / n, 1.0 / max(1, nopt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/work/kernel/e2e_results")
    ap.add_argument("--repo", default="google/gemma-4-12B-it")
    ap.add_argument("--json-out", default="/work/kernel/e2e_summary.json")
    args = ap.parse_args()

    data = load(args.out_dir, args.repo)
    if not data:
        print(f"no result files in {args.out_dir} for {args.repo}")
        return 1

    benches = sorted({b for b, _ in data})
    summary = {"repo": args.repo, "benchmarks": {}}
    # Every (benchmark, arm) test this run reports, for the family-wise correction below.
    fam: list[tuple[str, str, float]] = []

    for bench in benches:
        arms = [a for (b, a) in data if b == bench]
        arms = [a for a in ARM_ORDER if a in arms] + \
               sorted(a for a in arms if a not in ARM_ORDER)
        ref = data.get((bench, "bf16"))
        if not ref:
            print(f"\n### {bench}: no bf16 run, cannot compute paired deltas")
            continue
        # What this benchmark can and cannot resolve, printed before the numbers so the
        # table is read with its resolution in mind. GPQA Diamond is 198 items over 4
        # options: one arm's accuracy carries a ~3.4 point standard error, so a gap
        # between two arms needs roughly 9-10 points before it is anything but noise.
        # Ranking six arms inside a 7 point spread, as the earlier plot did, is reading a
        # signal that is not there.
        rrows = list(ref.values())
        nq = len(ref)
        nopt = max((r.get("n_options") or
                    len(r.get("option_scores") or r.get("letter_logprobs") or [1]))
                   for r in rrows)
        acc0 = sum(r["correct"] for r in rrows) / nq
        se = math.sqrt(acc0 * (1 - acc0) / nq)
        perms = rrows[0].get("perms", 1)
        print(f"\n### {bench}   (paired against bf16 on identical questions)")
        print(f"  {nq} items, {nopt} options (chance {1/nopt:.1%}), "
              f"{perms} option-order rotation(s) averaged")
        print(f"  bf16 is {acc0/(1/nopt):.2f}x chance; 1 s.e. on one arm is "
              f"{se*100:.1f} pt, so an arm-vs-arm gap needs ~{2.8*se*100:.1f} pt "
              f"to be distinguishable")
        if acc0 < 2.0 / nopt:
            print("  WARNING: bf16 is under 2x chance here, so most items are not "
                  "answered from knowledge and\n           differences between arms "
                  "largely reflect where each one's position prior fell")
        print(f"\n{'arm':10s} {'wbits':>5s} {'n':>5s} {'acc':>8s} {'delta':>8s} "
              f"{'95% CI':>17s} {'b/c':>9s} {'McNemar p':>10s} {'margin':>8s} "
              f"{'top slot':>9s}")
        summary["benchmarks"][bench] = {}
        summary["benchmarks"][bench]["_meta"] = {
            "n_items": nq, "n_options": nopt, "chance": 1.0 / nopt, "perms": perms,
            "bf16_acc": acc0, "se_one_arm": se,
            "min_resolvable_gap": 2.8 * se}

        for arm in arms:
            cur = data[(bench, arm)]
            common = sorted(set(ref) & set(cur))
            if not common:
                continue
            x = np.array([ref[q]["correct"] for q in common], dtype=float)
            y = np.array([cur[q]["correct"] for q in common], dtype=float)
            acc = y.mean()
            delta = acc - x.mean()
            margins = [m for m in (gold_margin(cur[q]) for q in common) if m is not None]
            mmean = float(np.mean(margins)) if margins else float("nan")
            top, unif = slot_bias([cur[q] for q in common])
            tflag = "!" if top > 2 * unif else " "

            if arm == "bf16":
                print(f"{arm:10s} {ARM_BITS.get(arm,''):5} {len(common):5d} "
                      f"{acc:8.4f} {'--':>8s} {'--':>17s} {'--':>9s} {'--':>10s} "
                      f"{mmean:8.4f} {top:8.0%}{tflag}")
                summary["benchmarks"][bench][arm] = {
                    "n": len(common), "acc": acc, "mean_margin": mmean,
                    "top_slot_frac": top}
                continue

            b = int(((x == 1) & (y == 0)).sum())
            c = int(((x == 0) & (y == 1)).sum())
            p = mcnemar_exact(b, c)
            lo, hi = paired_bootstrap(x, y)
            star = "*" if p < 0.05 else " "
            print(f"{arm:10s} {ARM_BITS.get(arm,''):5} {len(common):5d} "
                  f"{acc:8.4f} {delta:+8.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"{b:4d}/{c:<4d} {p:9.4f}{star} {mmean:8.4f} {top:8.0%}{tflag}")
            summary["benchmarks"][bench][arm] = {
                "n": len(common), "acc": acc, "delta_vs_bf16": delta,
                "ci95": [lo, hi], "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
                "mean_margin": mmean, "significant_at_05": bool(p < 0.05),
                "top_slot_frac": top}
            fam.append((bench, arm, p))

        print("  b = bf16 correct & arm wrong,  c = bf16 wrong & arm correct;  "
              "* = p<0.05")
        print("  margin = mean(score(chosen) - score(gold)); 0 is perfect, "
              "higher is worse")
        print(f"  top slot = share of answers on the single most-chosen option position "
              f"(uniform {unif:.0%});\n             '!' marks over twice uniform, i.e. "
              f"an arm answering from position rather than content")

    # Family-wise correction over every arm-vs-bf16 test above. 34 tests at 0.05 expect ~1.7
    # spurious rejections, and "this arm is measurably worse than bf16" is the sentence the
    # deck repeats most, so the per-test star is not the claim to publish. Holm rather than
    # Bonferroni: uniformly more powerful, no independence needed (all tests share bf16).
    # Non-rejections are unaffected -- correcting can only make a test harder to pass -- so
    # every "indistinguishable from bf16" statement stands as it was.
    if fam:
        ps = [p for _, _, p in fam]
        keep = holm(ps, 0.05)
        flipped = []
        for (bench, arm, p), k in zip(fam, keep):
            rec = summary["benchmarks"][bench][arm]
            rec["significant_holm"] = bool(k)
            rec["holm_n_tests"] = len(ps)
            if (p < 0.05) and not k:
                flipped.append((bench, arm, p))
        print(f"\n### family-wise correction: {sum(keep)} of {len(ps)} arm-vs-bf16 tests "
              f"survive Holm at 0.05")
        if flipped:
            print("  below 0.05 alone, not after correcting for having asked "
                  f"{len(ps)} times:")
            for bench, arm, p in flipped:
                print(f"    {bench:<9} {arm:<13} p={p:.4f}")
            print("  These are the cells where 'measurably worse than bf16' is not "
                  "supported.")

    print("\n### arm definitions")
    for a in _BASE_ORDER:
        if a in ARM_NOTE:
            print(f"  {a:10s} {ARM_BITS[a]}-bit weights   {ARM_NOTE[a]}")
    print("  NOTE: fp6 is the only arm the kernel does not block-scale -- it feeds the "
          "MFMA a\n        scale operand of 1.0 and applies one f32 per token/channel "
          "in the epilogue.\n        Its scale granularity is 32x coarser, which "
          "kernel/p15_fp6_scale_study.py measures at\n        3.1x the layer error of "
          "the same 6-bit elements block-scaled. That coarseness, not the\n        element "
          "width, is what put a 6-bit arm below every 4-bit one -- w6a6 is the same 6-bit"
          "\n        elements at per-32 E8M0 and is the arm to compare against the 4-bit "
          "ones.")
    print("  " + EQ_NOTE)

    with open(args.json_out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
