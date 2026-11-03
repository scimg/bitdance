#!/usr/bin/env python3
"""Full-model logit agreement with bf16, computed from the eval runs already on disk.

The plan asked for logit agreement on "a fixed prompt". The accuracy runs give something
strictly stronger for free: every arm scored the *same* 2198 prompts and recorded the
full answer-letter logprob vector for each, so agreement can be measured over thousands
of real prompts instead of one hand-picked one.

Accuracy alone is a coarse instrument here. Two arms can post the same score while
disagreeing with bf16 on different questions, and an arm can lose accuracy either by
perturbing many answers slightly or a few catastrophically. Three metrics separate those:

  top1_agree   how often the arm picks bf16's letter, right or wrong. Sensitive to
               perturbation regardless of whether it happened to help or hurt.
  flip_gain    of the answers that changed, the fraction that changed to correct.
               ~0.5 means the arm is adding noise; well below 0.5 means real damage.
  logit_mad    mean absolute difference in letter logprobs, in nats, after centering
               each question's vector on its own mean. Centering matters: an arm that
               shifts probability mass onto or off the letter tokens as a group moves
               every logprob together without changing which letter wins, and the raw
               figure counts that as damage when the answer never moved. Only the
               differences *between* letters can change an answer.

top1_agree is the gate: it is a paired, deterministic, per-question comparison against
bf16 on identical inputs, which is exactly what the layer gate was meant to establish.

A low top1_agree beside a near-zero accuracy delta is not a contradiction, but it is
not self-evidently benign either, so it is not left as a coincidence. The split by
bf16 correctness on the flipped and stable subsets decides between the two readings:
if bf16 scores near chance on the questions the arm flipped, the flips are landing on
ties the model was already guessing, and an unchanged score is the honest conclusion.
If bf16 scores well on them, the arm is destroying answers the model actually knew and
is being rescued by luck elsewhere -- same accuracy, entirely different verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="google/gemma-4-12B-it")
    ap.add_argument("--dir", default=os.path.join(_HERE, "e2e_results"))
    ap.add_argument("--arms", default="fp4,w4a6,fp6,w4a4_tc,w6a6_tc")
    ap.add_argument("--benches", default="gpqa,mmlu_pro")
    ap.add_argument("--out", default=os.path.join(_HERE, "e2e_agreement.json"))
    args = ap.parse_args()

    base = os.path.basename(args.repo)
    summary = {"model": base, "cells": {}}

    for bench in args.benches.split(","):
        ref = read(os.path.join(args.dir, f"{base}__{bench}__bf16.jsonl"))
        if not ref:
            print(f"{bench}: no bf16 reference, skipping")
            continue
        print(f"\n{base}  {bench}   (bf16 reference: {len(ref)} questions)")
        print(f"{'arm':10s}{'n':>6s}{'top1_agree':>12s}{'flips':>7s}"
              f"{'flip_gain':>11s}{'mad_ctr':>9s}{'mad_raw':>9s}{'acc':>8s}")
        for arm in args.arms.split(","):
            cur = read(os.path.join(args.dir, f"{base}__{bench}__{arm}.jsonl"))
            ids = sorted(set(ref) & set(cur))
            if not ids:
                continue
            agree = flips = flip_to_correct = ncorrect = 0
            ref_ok_flip = ref_ok_stay = 0
            marg_flip, marg_stay = [], []
            mad_c = mad_r = mad_n = 0.0
            for i in ids:
                a, b = ref[i], cur[i]
                flipped = a["choice"] != b["choice"]
                if flipped:
                    flips += 1
                    flip_to_correct += int(b["correct"])
                    ref_ok_flip += int(a["correct"])
                else:
                    agree += 1
                    ref_ok_stay += int(a["correct"])
                ncorrect += b["correct"]

                pa, pb = a.get("letter_logprobs"), b.get("letter_logprobs")
                if pa and pb and len(pa) == len(pb) and len(pa) > 1:
                    fa = [x for x in pa if math.isfinite(x)]
                    if len(fa) == len(pa):
                        # bf16's own decision margin: how close this question was to a
                        # tie before quantization touched it.
                        s = sorted(pa, reverse=True)
                        (marg_flip if flipped else marg_stay).append(s[0] - s[1])
                        ma = sum(pa) / len(pa)
                        mb = sum(pb) / len(pb)
                        for x, y in zip(pa, pb):
                            mad_r += abs(x - y)
                            mad_c += abs((x - ma) - (y - mb))
                            mad_n += 1
            n = len(ids)
            avg = lambda v: (sum(v) / len(v)) if v else None  # noqa: E731
            rec = {
                "n": n,
                "top1_agree": agree / n,
                "flips": flips,
                "flip_gain": (flip_to_correct / flips) if flips else None,
                "mad_centered": (mad_c / mad_n) if mad_n else None,
                "mad_raw": (mad_r / mad_n) if mad_n else None,
                "acc": ncorrect / n,
                # The diagnostic: was bf16 any good on the questions this arm flipped?
                "bf16_acc_on_flipped": (ref_ok_flip / flips) if flips else None,
                "bf16_acc_on_stable": (ref_ok_stay / agree) if agree else None,
                "bf16_margin_flipped": avg(marg_flip),
                "bf16_margin_stable": avg(marg_stay),
            }
            summary["cells"][f"{bench}/{arm}"] = rec
            fg = "      n/a" if rec["flip_gain"] is None else f"{rec['flip_gain']:11.3f}"
            print(f"{arm:10s}{n:6d}{rec['top1_agree']:12.4f}{flips:7d}"
                  f"{fg}{rec['mad_centered']:9.3f}{rec['mad_raw']:9.3f}"
                  f"{rec['acc']:8.4f}")

        print(f"\n  where the flips land ({bench})")
        print(f"  {'arm':10s}{'bf16 acc / flipped':>20s}{'bf16 acc / stable':>20s}"
              f"{'bf16 margin flip':>18s}{'stable':>9s}")
        for arm in args.arms.split(","):
            r = summary["cells"].get(f"{bench}/{arm}")
            if not r or r["bf16_acc_on_flipped"] is None:
                continue
            print(f"  {arm:10s}{r['bf16_acc_on_flipped']:20.3f}"
                  f"{r['bf16_acc_on_stable']:20.3f}"
                  f"{r['bf16_margin_flipped']:18.3f}"
                  f"{r['bf16_margin_stable']:9.3f}")

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwrote {args.out}")
    print("\ntop1_agree is agreement with bf16's answer, not correctness. An arm can")
    print("hold its accuracy while disagreeing often; that shows up here and nowhere else.")


if __name__ == "__main__":
    main()
