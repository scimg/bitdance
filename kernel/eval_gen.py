#!/usr/bin/env python3
"""Generative benchmark scoring: the protocol the model card actually used.

WHY THIS EXISTS. The 31B card reports 85.2% on MMLU-Pro and 84.3% on GPQA Diamond.
kernel/eval_bench.py measures 72.3% and 53.0% for the same weights. That 31-point GPQA
gap is not quantization and not a scoring bug -- it is a different task. eval_bench
reads log_softmax(logits[-1]) at the answer-letter ids on ONE forward pass, so it asks a
reasoning model to answer a graduate physics question in a single token with no working.
The card's numbers come from the model's thinking mode; its comparison column is
labelled "Gemma 3 27B (no think)".

Worse, and this is the part that was invisible: the canonical Gemma 4 chat template
appends `<|channel>thought\\n<channel|>` -- an EMPTY thought block -- when
`enable_thinking` is not passed. Per the card that is precisely how thinking is
DISABLED. So every accuracy cell in this repo was recorded with the model's reasoning
mode switched off, by a template default nobody passed an argument to.

    no-think (what eval_bench renders, implicitly):
        <bos><|turn>user\\n...<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>
    think (what the card used):
        <bos><|turn>system\\n<|think|>\\n<turn|>\\n<|turn>user\\n...<turn|>\\n<|turn>model\\n

WHY IT ALSO FIXES THE MEASUREMENT PROBLEM. ACCURACY_DEBUG.md section 10 shows raw
accuracy is fidelity plus a damage-reward channel worth (1 - acc_bf16)/(k - 1): an arm
scores by departing from bf16 and landing right. At bf16 = 42.4% on 4-option GPQA that
channel is 19% of items and it pays the WORSE arm more, which is what put w4a6 under
w4a4. At the card's 84.3% it is 5.2%, and the fidelity slope goes from 0.13 to ~0.79
accuracy points per point of agreement. Restricting the score to a subset was the wrong
fix for this; running the benchmark the way it was meant to be run is the right one,
because it shrinks the luck channel on the WHOLE dataset instead of deleting items.

DECODING. The card's temperature=1.0/top_p=0.95/top_k=64 is the default here, and greedy
is available but should not be used: measured, greedy makes this model ramble. On 12B
GPQA with thinking on, greedy truncated 31 of 32 items at 6,144 tokens with a median of
exactly max_new -- the traces re-examine the same chiral centre forever and never reach
an answer, so accuracy came out at 9.4% with 20 of 32 replies unparseable. That is a
decoding failure, not a model or a quantization result, and it would have been charged
to whichever arm was unlucky enough to be measured with it.

Sampling costs run-to-run variance, which is real and is the price of the protocol. Pay
it down with --repeat k rather than by switching decoders: k independent samples per
item, scored as mean pass@1, divides the decoding component of the variance by k.

    python3 -m kernel.eval_gen --bench gpqa --arm bf16 --repo google/gemma-4-12B-it
"""
import argparse
import json
import os
import re
import time

from kernel.eval_bench import LETTERS, load_bench

INSTRUCTION = ("Answer the following multiple choice question. Reason it through, then "
               "finish your reply with a final line of exactly the form "
               "'The answer is (X)' where X is the letter of the correct option.")

# The template's own turn terminator. Generation stops here; without it the model runs on
# into a fresh turn and the parser sees a second, unrelated answer.
STOP = "<turn|>"
# Thinking output is `<|channel>thought\n ...reasoning... <channel|>` then the reply. The
# reasoning routinely names options it goes on to reject, so parsing must happen AFTER the
# last close tag or a rejected candidate gets scored as the answer.
THOUGHT_CLOSE = "<channel|>"


def build_prompt(tok, it, think=True):
    body = [f"Question: {it['stem']}", "Options:"]
    body += [f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"])]
    msgs = [{"role": "user", "content": "\n".join([INSTRUCTION, ""] + body)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=think)


def extract(text, n_opt):
    """The answer letter, or None. Returns (letter_index, how_it_was_found).

    Ordered by how much the model committed to the answer, and every pattern takes its
    LAST match: a reply that revisits its choice should be scored on where it landed.
    """
    tail = text.rsplit(THOUGHT_CLOSE, 1)[-1] if THOUGHT_CLOSE in text else text
    tail = tail.split(STOP)[0]
    valid = LETTERS[:n_opt]
    pats = [("answer_is", r"answer\s+is\s*[:\-]?\s*\(?\s*([A-Za-z])\s*\)?"),
            ("bold", r"\*\*\s*\(?\s*([A-Za-z])\s*\)?\s*\*\*"),
            ("paren", r"\(\s*([A-Za-z])\s*\)"),
            ("line", r"(?m)^\s*\(?([A-Za-z])[\)\.\:]\s")]
    for how, p in pats:
        m = [x for x in re.finditer(p, tail) if x.group(1).upper() in valid]
        if m:
            return valid.index(m[-1].group(1).upper()), how
    # Last resort, and only if the reply is short enough that a bare letter is the answer
    # rather than a letter mentioned in passing.
    if len(tail.strip()) <= 24:
        m = [c for c in tail.upper() if c in valid]
        if m:
            return valid.index(m[-1]), "bare"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--repo", default="google/gemma-4-12B-it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-dir", default="/work/kernel/e2e_gen")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-think", action="store_true",
                    help="render the way eval_bench implicitly did, for the comparison "
                         "that shows thinking is the whole gap to the card")
    ap.add_argument("--greedy", action="store_true",
                    help="do not use: measured, this model rambles to max_new on 31 of "
                         "32 GPQA items under greedy. Kept so that claim is reproducible")
    ap.add_argument("--repeat", type=int, default=1,
                    help="independent samples per item, scored as mean pass@1. Divides "
                         "the decoding variance by k without changing the protocol")
    ap.add_argument("--torch-seed", type=int, default=1234,
                    help="sampling seed, so a cell is reproducible run to run. Distinct "
                         "from --seed, which picks the item subset")
    ap.add_argument("--shard", default="1/1",
                    help="'i/n': score only every n-th item, offset i-1, into a cell tagged "
                         "'...__shardIofN'. For the 12,032-item MMLU-Pro run, where one card is "
                         "34 h for the CHEAPEST arm -- see the note in main()")
    ap.add_argument("--carry-from", default="",
                    help="seed this cell with the records of an earlier one that were NOT "
                         "truncated, leaving only the capped items to regenerate. The way to "
                         "ask what a larger --max-new is worth: see the note in main()")
    a = ap.parse_args()

    import torch
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.tag or a.arm

    # SHARDING IS A SCHEDULE, NOT A PROTOCOL. `--shard i/n` slices the pool with a stride, so the n
    # shards are disjoint and their union is exactly the pool an unsharded run would have scored --
    # kernel/p212_merge.py concatenates them back into one cell. It exists because the whole-split
    # MMLU-Pro run does not fit on one card in any useful time: the s38 logs put bf16 at 84 min for
    # 500 items, which is 34 h for 12,032, and bf16 is the CHEAPEST arm. `w4a4` generates 5.5x as
    # many tokens per item, so unsharded it is over a week.
    #
    # What it does NOT preserve is the draw. Sampling is seeded once per process and consumed in
    # batch order, so item k in shard 3 of 8 sees different randomness than item k would have in an
    # unsharded run. That is the same caveat every re-run in this tree carries (ACCURACY_DEBUG.md
    # §16.7) and it is why the item set, not the token stream, is what gets compared across cells.
    try:
        shard_i, shard_n = (int(v) for v in a.shard.split("/"))
    except Exception:
        raise SystemExit(f"--shard must look like 'i/n', got {a.shard!r}")
    if not (1 <= shard_i <= shard_n):
        raise SystemExit(f"--shard {a.shard}: need 1 <= i <= n")
    if shard_n > 1:
        name = f"{name}__shard{shard_i}of{shard_n}"

    tag = f"{os.path.basename(a.repo)}__{a.bench}__{name}"
    out_path = os.path.join(a.out_dir, tag + ".jsonl")

    # WHAT IS A LARGER max_new WORTH? The honest way to ask is to change the cap and nothing
    # else, and a plain re-run cannot do that: sampling at T=1.0 means every item comes back
    # different, so the answer arrives buried in decoding noise that is the same size as the
    # effect. Carrying the untruncated records forward verbatim leaves the cap as the ONLY
    # difference between the two cells -- the items it never bound are literally the same
    # records, and the items it cut are regenerated with more room. What comes out is a paired
    # comparison over exactly the items in question.
    #
    # It is also most of the cost saved: on 12B GPQA the cap binds 59% of w4a4's items and 2%
    # of bf16's, so a full re-run would spend the bulk of its time resampling items that had
    # already finished well inside the old budget.
    #
    # Only when the new cell does not exist yet. A resumed run must not re-carry, or an item
    # regenerated at the new cap and truncated AGAIN would be replaced by its old record and
    # the cell would silently stop converging.
    carried_ids = None
    if a.carry_from:
        src_recs = []
        with open(a.carry_from) as src:
            for line in src:
                try:
                    src_recs.append(json.loads(line))
                except Exception:
                    pass
        # Read on EVERY run, not only the first, because it is what pins the item set below and
        # a resumed run needs the same pin. Only the file writing is first-run-only.
        carried_ids = {r["id"] for r in src_recs}
        if not os.path.exists(out_path):
            keep = [r for r in src_recs if not r.get("truncated")]
            with open(out_path, "w") as out:
                for r in keep:
                    out.write(json.dumps(r) + "\n")
            print(f"carried {len(keep)} untruncated records from "
                  f"{os.path.basename(a.carry_from)}; {len(src_recs) - len(keep)} capped items "
                  f"to regenerate at max_new={a.max_new}", flush=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["id"], r.get("rep", 0)))
                except Exception:
                    pass
    pool = load_bench(a.bench, a.limit, a.seed)

    # THE CARRY SOURCE DEFINES THE ITEM SET, not --limit. A cell being extended has to cover
    # exactly the items its source covered, and leaving that to a flag is how this ran the whole
    # 12,032-item MMLU-Pro instead of the 500-item subset the cell it was extending used: the
    # carried 500 counted as done, the other 11,532 counted as work, and four cards quietly
    # signed up for four days on an item set nothing would ever compare against. --limit is not
    # a description of the cell, it is a description of one run.
    if a.carry_from and carried_ids is not None:
        keep = [it for it in pool if it["id"] in carried_ids]
        if len(keep) != len(carried_ids):
            print(f"  WARNING: carry source has {len(carried_ids)} items, "
                  f"{len(keep)} of them are in this bench's pool", flush=True)
        if keep:
            pool = keep

    # Sliced AFTER the carry-source pin above, so a shard of an extended cell still covers only the
    # items its source covered.
    if shard_n > 1:
        pool = pool[shard_i - 1::shard_n]

    items = [dict(it, rep=k) for k in range(a.repeat) for it in pool
             if (it["id"], k) not in done]
    print(f"{tag}: {len(items)} runs to go ({len(done)} done) over {len(pool)} items "
          f"x{a.repeat}, think={not a.no_think}, greedy={a.greedy}, "
          f"batch={a.batch}, max_new={a.max_new}", flush=True)
    if not items:
        return 0

    from kernel import gemma_quant as gq
    if a.arm == "bf16":
        model, tok = gq.load_model(a.repo, device="cuda", device_map=a.device_map)
    else:
        model, tok, rep = gq.load_quantized(a.repo, a.arm, device="cuda", verbose=True)
        print(f"  patched {rep['replaced']} linears; skipped={rep['skipped']}", flush=True)
        if rep["skipped"]:
            print("  WARNING: skipped layers stay bf16 and will flatter this arm")
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # DO NOT pass eos_token_id. The model's generation_config carries a LIST of three
    # terminators ([1, 106, 50]); passing tok.eos_token_id replaces all three with one,
    # the turn terminator stops being a stop, and every item runs to max_new. That is
    # what the first run of this file did -- 16 of 16 truncated at 4096 tokens, 0%.
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    gen = dict(max_new_tokens=a.max_new, do_sample=not a.greedy, pad_token_id=pad)
    gen.update(dict(temperature=None, top_p=None, top_k=None) if a.greedy
               else dict(temperature=1.0, top_p=0.95, top_k=64))
    torch.manual_seed(a.torch_seed)

    # Length-sorted batches: a batch is as slow as its longest member, and GPQA stems vary
    # by an order of magnitude. Written back out in id order by the reader, so this only
    # affects speed.
    items.sort(key=lambda it: len(it["stem"]) + sum(len(o) for o in it["options"]))
    t0, n_ok, n_seen, n_cut, n_bad = time.time(), 0, 0, 0, 0
    with open(out_path, "a") as fh:
        for s in range(0, len(items), a.batch):
            chunk = items[s:s + a.batch]
            prompts = [build_prompt(tok, it, think=not a.no_think) for it in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, **gen)
            plen = enc["input_ids"].shape[1]
            for it, row in zip(chunk, out):
                new = row[plen:]
                keep = int((new != pad).sum())
                # Trim the right padding before decoding: otherwise the stored tail is
                # 400 characters of "<pad>" and useless for auditing a parse by eye.
                txt = tok.decode(new[:keep] if keep else new, skip_special_tokens=False)
                ntok = keep
                cut = ntok >= a.max_new - 2
                pred, how = extract(txt, len(it["options"]))
                ok = pred is not None and pred == it["gold_idx"]
                n_seen += 1
                n_ok += int(ok)
                n_cut += int(cut)
                n_bad += int(pred is None)
                fh.write(json.dumps({
                    "id": it["id"], "rep": it.get("rep", 0),
                    "category": it.get("category", ""),
                    "gold_idx": it["gold_idx"], "pred_idx": pred, "correct": bool(ok),
                    "parsed": pred is not None, "how": how, "truncated": bool(cut),
                    "n_gen": ntok, "n_options": len(it["options"]),
                    # Enough tail to audit a parse by eye without storing whole traces.
                    "tail": txt[-400:]}) + "\n")
            fh.flush()
            el = time.time() - t0
            print(f"  {n_seen}/{len(items)}  acc={100.0 * n_ok / n_seen:.1f}%  "
                  f"unparsed={n_bad}  truncated={n_cut}  "
                  f"{el / 60:.1f} min  eta {el / n_seen * (len(items) - n_seen) / 60:.0f} min",
                  flush=True)
    print(f"\n{tag}: {100.0 * n_ok / max(n_seen, 1):.2f}%  "
          f"({n_ok}/{n_seen}), unparsed {n_bad}, truncated {n_cut}, "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
