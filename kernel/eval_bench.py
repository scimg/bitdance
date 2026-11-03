#!/usr/bin/env python3
"""Accuracy benchmark harness for the quantization arms.

Scoring is cloze-style: build the chat prompt, run ONE forward pass per option ordering,
and compare the logits of the candidate answer-letter tokens at the final position. Both
benchmarks present their options inline and expect a single letter, so this reads the
model's answer directly rather than generating and parsing it.

Why not avg@8 chain-of-thought generation, as originally planned: 6 variants x 8 samples
x ~500 generated tokens is ~50x the compute of one forward pass per question, and this
node cannot hold a GPU for that long (see supervise.sh). Cloze scoring also makes the
comparison *better* for the question being asked, which is how much accuracy quantization
costs: it is deterministic, so a paired per-question comparison against bf16 has no
sampling noise at all, and every arm sees byte-identical prompts.

The cost is that the absolute numbers are cloze accuracy, NOT the published CoT accuracy,
so they will not reproduce Gemma's reported 78.8% on GPQA Diamond.

TWO THINGS THIS FIXES, both found by kernel/p15_bench_audit.py after GPQA and MMLU-Pro
ranked the arms in contradictory orders (fp4 best on one, worst by 11 points on the other):

  * OPTION-ORDER BIAS. On GPQA the bf16 model answered "A" for 67% of 198 items, against a
    25% base rate, with a prediction entropy of 1.44 bits out of a possible 2.00. At
    1.2-1.5x chance the model mostly does not know the answer, and what it emits then is
    its prior over letters, not a judgement about the options. Accuracy becomes a question
    of how well that prior happens to line up with where the gold letter sits, and
    quantization perturbs the prior in whichever direction it likes. `--perms` scores each
    item under several cyclic rotations of the option order and averages each option's
    logprob across the slots it occupied, which cancels the position prior. For a 4-option
    benchmark 4 rotations cancels it exactly.

  * THE MODEL WAS NOT ASKED FOR A LETTER (`--protocol`, added later). Reading the letter
    logits only measures the model if a letter is what it was about to emit, and mostly it
    was not: kernel/p110_acc_audit.py puts 12B bf16 at 0.373 mean letter mass on MMLU-Pro
    and 0.009 on GPQA, so 63% and 99% of items were scored off the tail of a distribution
    about something else. kernel/p142_letter_mass.py shows the model writing prose -- the
    top token at the decision position is 'To' at p=0.87. That is a benchmark spending most
    of its n on noise, and it is the reason nearly every ordering in p129_orderings.py came
    back "level" rather than resolving. See PROTOCOLS below; `plain` is the original and
    remains the default so no recorded cell changes meaning.

  * PROMPT FORMAT SKEW. MMLU-Pro was given a structured "Options:" block built here, while
    GPQA was passed through with whatever formatting its rows happened to carry, on the
    strength of a comment saying the options were already inline. They are, but in two
    different layouts (kernel/p15_gpqa_probe.py: 197 items label the options A-D directly,
    1 stacks a lower-case list under a shuffled A->a map). Both benchmarks are now parsed
    into (stem, options, gold index) and rendered by one function, so a cross-benchmark
    comparison is not also a comparison of two prompt formats.

Resumable by construction: one JSONL line per question, and completed question ids are
skipped on restart. The node kills this container on a timescale shorter than a full run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HOME", "/work/hf_cache")
sys.path.insert(0, "/root")
sys.path.insert(0, "/work")

import torch

LETTERS = "ABCDEFGHIJ"

BENCHES = {
    # name: (repo, config, split)
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
    "gpqa": ("fingertap/GPQA-Diamond", None, "test"),
}


# ---- GPQA option recovery ---------------------------------------------------- #
#
# The rows carry the options inside the question text. Recovering them as a list is what
# lets the order be rotated for debiasing and lets the prompt be rendered in the same
# format as MMLU-Pro's.

_OPT_UPPER = re.compile(
    r"(?:^|\n)[ \t]*([A-D])[\.\)][ \t]*(.*?)(?=\n[ \t]*[A-D][\.\)]|\Z)", re.S)
_OPT_LOWER = re.compile(
    r"(?:^|\n)[ \t]*([a-d])[\.\)][ \t]*(.*?)(?=\n[ \t]*[a-d][\.\)]|\n[ \t]*[A-D][\.\)]|\Z)",
    re.S)


def _clean(s):
    return " ".join(s.split())


def parse_gpqa(question, gold_letter):
    """(stem, [4 option texts], gold index) from a GPQA row.

    Two layouts occur. Most rows label the four options A-D directly. One stacks a
    lower-case a)-d) list and then maps A->d, B->a and so on, which turns the task into
    carrying a symbol through a lookup; resolving the map here removes that.
    """
    up = [(m.group(1), m.group(2)) for m in _OPT_UPPER.finditer(question)]
    if len(up) != 4 or [l for l, _ in up] != list("ABCD"):
        raise ValueError("no A-D option block")
    texts = [_clean(t) for _, t in up]
    gi = "ABCD".index(gold_letter)

    head = question[: _OPT_UPPER.search(question).start()]

    # Indirection: every "option" is just a lower-case letter naming a real option, which
    # lives in an a)-d) list above. Resolve the map, and cut the list out of the stem --
    # leaving it in would show every option twice, once unlabelled and once labelled.
    if all(t.lower() in ("a", "b", "c", "d") for t in texts):
        low = {m.group(1): _clean(m.group(2)) for m in _OPT_LOWER.finditer(question)}
        if set(low) != set("abcd"):
            raise ValueError("indirection without a complete a)-d) list")
        texts = [low[t.lower()] for t in texts]
        first_low = _OPT_LOWER.search(head)
        if first_low:
            head = head[: first_low.start()]

    stem = _clean(head)
    if not stem:
        raise ValueError("empty stem")
    return stem, texts, gi


# ---- dataset normalisation --------------------------------------------------- #

def load_bench(name, limit=None, seed=0, split=None):
    """Return a list of {id, stem, options, gold_idx, category}.

    `split` overrides the scored split. Only the quantizer's calibration uses it, to build
    its activation statistics from MMLU-Pro's `validation` items -- same format and
    categories as the scored ones, disjoint from them.
    """
    from datasets import load_dataset
    repo, config, dflt = BENCHES[name]
    split = split or dflt
    ds = load_dataset(repo, config, split=split) if config else \
        load_dataset(repo, split=split)

    items = []
    if name == "mmlu_pro":
        for r in ds:
            gold = r["answer"].strip().upper()
            items.append({"id": str(r["question_id"]), "stem": _clean(r["question"]),
                          "options": [_clean(str(o)) for o in r["options"]],
                          "gold_idx": LETTERS.index(gold),
                          "category": r["category"]})
    else:
        bad = []
        for i, r in enumerate(ds):
            try:
                stem, opts, gi = parse_gpqa(r["question"], r["answer"].strip().upper())
            except ValueError as e:
                bad.append((i, str(e)))
                continue
            items.append({"id": str(i), "stem": stem, "options": opts, "gold_idx": gi,
                          "category": r.get("domain", "")})
        if bad:
            # Loud, not silent: dropping items would change n between arms if the parse
            # were ever nondeterministic, and a mixture of formats is what this replaced.
            raise RuntimeError(f"{len(bad)} GPQA rows did not parse: {bad[:5]}")

    if limit and limit < len(items):
        # Stratify over categories so a subsample cannot accidentally reweight the
        # benchmark towards its easy or hard subjects.
        import random
        from collections import defaultdict
        rng = random.Random(seed)
        by_cat = defaultdict(list)
        for it in items:
            by_cat[it["category"]].append(it)
        for v in by_cat.values():
            rng.shuffle(v)
        cats = sorted(by_cat)
        picked, i = [], 0
        while len(picked) < limit:
            progressed = False
            for c in cats:
                if i < len(by_cat[c]) and len(picked) < limit:
                    picked.append(by_cat[c][i])
                    progressed = True
            if not progressed:
                break
            i += 1
        items = sorted(picked, key=lambda x: x["id"])
    return items


# ---- scoring protocol --------------------------------------------------------- #
#
# The scorer reads log_softmax(logits[-1]) at the answer-letter token ids, so it is only
# measuring the model if the model was about to emit a bare letter there. Under `plain` --
# everything recorded before this existed -- it mostly was not: kernel/p110_acc_audit.py
# puts 12B bf16's mean letter mass at 0.373 on MMLU-Pro and 0.009 on GPQA, and
# kernel/p142_letter_mass.py shows why, with the top token at the decision position being
# 'To' at p=0.87. The model answers in prose. Every arm's accuracy was then a ranking in
# the tail of a distribution about something else, which is a measurement with most of its
# n spent on noise, and it is the reason nearly every ordering in kernel/p129_orderings.py
# comes back "level".
#
# A protocol is (assistant prefix appended after the generation prompt, letter format). The
# format matters as much as the prefix: after a prefix ending in a space the model's natural
# continuation is the SPACE-PREFIXED " A", a different token id, and reading bare "A" there
# reports a mass collapse that is arithmetic rather than behaviour.
#
# `plain` stays the default so no existing cell changes meaning. Pick a protocol with
# kernel/p144_protocol.py, which requires a candidate to raise letter mass AND accuracy --
# mass alone is buyable with a prefix so leading the model is answering something easier.
PROTOCOLS = {
    "plain": ("", "{}"),
    "paren": ("(", "{}"),
    "answer_is": ("The answer is ", " {}"),
    "answer_is_paren": ("The answer is (", "{}"),
    "answer_colon": ("Answer: ", " {}"),
}


def letter_token_ids(tok, n, protocol="plain"):
    """Token id of each answer letter as it appears at the start of the reply."""
    fmt = PROTOCOLS[protocol][1]
    ids = []
    for ch in LETTERS[:n]:
        enc = tok.encode(fmt.format(ch), add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"letter {fmt.format(ch)!r} is not a single token: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"answer letters collide in the vocabulary: {ids}")
    return ids


INSTRUCTION = ("Answer the following multiple choice question. "
               "Respond with only the letter of the correct option.")


def build_prompt(tok, stem, options, protocol="plain"):
    """One rendering for both benchmarks, so a cross-benchmark comparison is not also a
    comparison of two prompt layouts."""
    body = [f"Question: {stem}", "Options:"]
    body += [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    msgs = [{"role": "user", "content": "\n".join([INSTRUCTION, ""] + body
                                                  + ["", "Answer:"])}]
    return (tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            + PROTOCOLS[protocol][0])


def score_item(model, tok, it, perms, dev, max_len, protocol="plain"):
    """Mean logprob per option across `perms` cyclic rotations of the option order.

    Rotation r puts option i in slot (i + r) mod n, so averaging over r scores every
    option in a spread of positions and the model's preference for a *position* cancels
    instead of being credited to whichever option happened to sit there.
    """
    opts = it["options"]
    n = len(opts)
    lids = letter_token_ids(tok, n, protocol)
    totals = torch.zeros(n, dtype=torch.float64)
    per_rot = []
    for r in range(perms):
        # slot j shows option (j - r) mod n
        order = [(j - r) % n for j in range(n)]
        prompt = build_prompt(tok, it["stem"], [opts[i] for i in order], protocol)
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_len).to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)[lids].double().cpu()
        for j, i in enumerate(order):
            totals[i] += lp[j]
        per_rot.append([round(float(v), 5) for v in lp.tolist()])
    return (totals / perms), per_rot


# ---- run ---------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=list(BENCHES), required=True)
    ap.add_argument("--arm", required=True,
                    help="bf16 | fp4 | w4a6 | fp6 | w6a6_tc | w4a4_tc")
    ap.add_argument("--repo", default="google/gemma-4-12B-it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--out-dir", default="/work/kernel/e2e_results")
    ap.add_argument("--device-map", default=None,
                    help="'auto' shards bf16 across visible cards (needed for 31B)")
    ap.add_argument("--perms", type=int, default=4,
                    help="cyclic rotations of the option order to average over; 1 "
                         "reproduces the original biased scoring")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="plain",
                    help="how the reply is elicited; see PROTOCOLS. 'plain' is every cell "
                         "recorded before this flag existed and stays the default, but it "
                         "leaves 12B bf16 at 0.373 mean letter mass on MMLU-Pro -- i.e. "
                         "most of the item set is scored off the tail. A results directory "
                         "must not mix protocols; use --tag or a fresh --out-dir.")
    ap.add_argument("--smooth", default=None,
                    help="path to the activation amax file from p15_calibrate.py; "
                         "enables per-input-channel equalization")
    ap.add_argument("--tag", default=None,
                    help="output name for the arm; use when a config change makes the "
                         "results a different arm rather than an update of one "
                         "(e.g. w6a6_tc_g for the per-tensor global scale)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{os.path.basename(args.repo)}__{args.bench}__{args.tag or args.arm}"
    out_path = os.path.join(args.out_dir, tag + ".jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    items = load_bench(args.bench, args.limit, args.seed)
    todo = [it for it in items if it["id"] not in done]
    print(f"{tag}: {len(items)} questions, {len(done)} already done, "
          f"{len(todo)} to go, perms={args.perms}, protocol={args.protocol}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    from kernel import gemma_quant as gq
    from kernel import qsmooth as qs
    dev = "cuda"
    t0 = time.time()
    if args.arm == "bf16":
        model, tok = gq.load_model(args.repo, device=dev,
                                   device_map=args.device_map)
    else:
        amax = qs.load(args.smooth) if args.smooth else None
        if args.smooth and not amax:
            print(f"  WARNING: --smooth {args.smooth} has no data; running unequalized")
        # CPU-staged quantization: peak GPU memory is the quantized model, not the
        # bf16 one, which is what lets this run in the ~19 GiB the node usually leaves.
        model, tok, rep = gq.load_quantized(args.repo, args.arm, device=dev,
                                            verbose=True, act_amax=amax)
        print(f"  patched {rep['replaced']} linears; skipped={rep['skipped']}",
              flush=True)
        if rep["skipped"]:
            print("  WARNING: skipped layers stay bf16 and will flatter this arm")
        if amax:
            print(f"  equalized {rep['smoothed']}/{rep['replaced']} linears")
            if rep["unsmoothed"]:
                print(f"  WARNING: no calibration for {len(rep['unsmoothed'])} layers, "
                      f"e.g. {rep['unsmoothed'][:3]}")
        if rep["unknown_shapes"]:
            print(f"  untuned shapes (default tile): {rep['unknown_shapes']}")
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"  model ready in {time.time()-t0:.0f}s; {free:.1f} GiB VRAM free",
          flush=True)

    n_correct, n_seen = 0, 0
    t0 = time.time()
    with open(out_path, "a") as fh:
        for i, it in enumerate(todo):
            scores, per_rot = score_item(model, tok, it, args.perms, dev, args.max_len,
                                         args.protocol)
            pred = int(scores.argmax().item())
            correct = int(pred == it["gold_idx"])
            n_correct += correct
            n_seen += 1
            fh.write(json.dumps({
                "id": it["id"], "category": it["category"],
                "gold_idx": it["gold_idx"], "pred_idx": pred, "correct": correct,
                "n_options": len(it["options"]), "perms": args.perms,
                # On the row, not just in the log: rot_logprobs mean different things under
                # different protocols, and an analysis that pools two of them is comparing
                # prompt layouts. p110/p111 read this field.
                "protocol": args.protocol,
                # Per-option scores after debiasing: lets the paired analysis use margins
                # and lets a later metric be computed without re-running the model.
                "option_scores": [round(float(v), 5) for v in scores.tolist()],
                # Slot-indexed logprobs per rotation, which is what a later audit of
                # position bias needs and what the averaging above consumes.
                "rot_logprobs": per_rot,
                # Back-compatible fields so the existing analysis keeps working. With
                # perms>1 the "letter" is the slot the chosen option sat in at rotation 0.
                "gold": LETTERS[it["gold_idx"]], "choice": LETTERS[pred],
                "letter_logprobs": [round(float(v), 5) for v in scores.tolist()],
            }) + "\n")
            if (i + 1) % 25 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)}  acc={n_correct/n_seen:.4f}  "
                      f"{el/(i+1):.2f}s/q  eta={(len(todo)-i-1)*el/(i+1)/60:.1f}min",
                      flush=True)

    print(f"DONE {tag}: {n_correct}/{n_seen} = {n_correct/max(1,n_seen):.4f} "
          f"on the {n_seen} newly scored", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
