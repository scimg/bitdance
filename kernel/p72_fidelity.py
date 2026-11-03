#!/usr/bin/env python3
"""Agreement with bf16 at the output distribution -- the metric SNR is trying to be.

kernel/p68_ppl.py was meant to settle whether four features that improve reconstruction and
cost MMLU-Pro points are damaging the model or tripping the eval. It could not, and the
reason is worth writing down rather than hiding: wikitext was unreachable on this node, so it
fell back to chat-templated MMLU-Pro prompts, and perplexity over a template is not a quality
signal. Those prompts are hugely repetitive ("Options:", "A.", "B."), and a model that has
been pushed toward local copying scores WELL on them. Every quantized arm came out 20-31%
BELOW bf16, which is impossible as a quality statement and is a clean symptom of that bias.

This measures fidelity to bf16 directly, which needs no assumption about the text at all:

    top1     how often the quantized model's argmax next token equals bf16's
    nll_ref  the quantized model's NLL of the token BF16 would have predicted
    kl       KL(bf16 || quantized) on bf16's top-128 support, renormalised

All three are properties of the two distributions, not of the corpus, so the templating that
broke perplexity is harmless here. And they are exactly what a layer-SNR improvement claims
to deliver: if a feature raises SNR against bf16 at every layer, it must raise agreement with
bf16 at the output, or the SNR was measuring something that does not reach the output.

So the paradox becomes falsifiable in one shot:

    agreement UP while MMLU-Pro is DOWN   ->  the model is more faithful to bf16 and the
                                              eval is where the points are going
    agreement DOWN                        ->  the features damage the model, layer SNR
                                              never predicted the output, and four sessions
                                              of reconstruction audits measured the wrong
                                              thing

Two passes. `--dump` writes bf16's argmax and top-128 support once; every other config reads
it. Resumable per sequence, because this node kills GPU work about 100 seconds in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/root")
sys.path.insert(0, "/work")

import torch

TOPK = 128


def sequences(tok, n: int, seed: int):
    from kernel.eval_bench import build_prompt, load_bench
    items = load_bench("mmlu_pro", limit=n, seed=seed + 7919, split="validation")
    return [build_prompt(tok, it["stem"], it["options"]) for it in items]


def letter_ids(tok, n: int = 10):
    """The option-letter token ids the eval scores, in the eval's own order."""
    from kernel.eval_bench import letter_token_ids
    return list(letter_token_ids(tok, n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smooth", default=None)
    ap.add_argument("--repo", default="google/gemma-4-12B-it")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", default="/work/kernel/ppl_results/ref_bf16.pt")
    ap.add_argument("--dump", action="store_true", help="write the bf16 reference")
    ap.add_argument("--out-dir", default="/work/kernel/fid_results")
    a = ap.parse_args()

    tag = a.tag or a.arm
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.ref), exist_ok=True)

    from kernel import gemma_quant as gq
    from kernel import qsmooth as qs
    dev = "cuda"

    if a.dump:
        ref = torch.load(a.ref) if os.path.exists(a.ref) else {}
        model, tok = gq.load_model(a.repo, device=dev)
        seqs = sequences(tok, a.n, a.seed)
        letters = torch.tensor(letter_ids(tok), device=dev)
        with torch.no_grad():
            for i, t in enumerate(seqs):
                if i in ref:
                    continue
                ids = tok(t, return_tensors="pt", truncation=True,
                          max_length=a.max_len).input_ids.to(dev)
                full = model(ids).logits.float()
                lg = full[0, :-1]
                lp = torch.log_softmax(lg, dim=-1)
                v, ix = lp.topk(TOPK, dim=-1)
                # The ANSWER position: the eval reads exactly this one distribution,
                # restricted to the option letters. Agreement averaged over prompt tokens
                # can be excellent while this is not, and only this one scores the bench.
                alp = torch.log_softmax(full[0, -1], dim=-1)[letters]
                ref[i] = {"ids": ids.cpu(), "top_lp": v.cpu(), "top_ix": ix.cpu(),
                          "ans_lp": (alp - alp.logsumexp(0)).cpu()}
                torch.save(ref, a.ref)
                print(f"    ref {len(ref)}/{a.n}", flush=True)
        print(f"  bf16 reference: {len(ref)} sequences -> {a.ref}")
        return 0

    if not os.path.exists(a.ref):
        print(f"  no bf16 reference at {a.ref}; run with --arm bf16 --dump first")
        return 1
    ref = torch.load(a.ref)

    out = os.path.join(a.out_dir, f"fid__{tag}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["i"])
    todo = [i for i in sorted(ref) if i not in done]
    if not todo:
        print(f"  {tag}: already complete at {len(done)}")
        return 0

    amax = qs.load(a.smooth) if a.smooth else None
    model, tok, rep = gq.load_quantized(a.repo, a.arm, device=dev, verbose=False,
                                        act_amax=amax)
    print(f"  {tag}: patched {rep['replaced']}, equalized "
          f"{rep['smoothed'] if amax else 0}; {len(todo)} sequences to go", flush=True)

    fh = open(out, "a", encoding="utf-8")
    with torch.no_grad():
        for i in todo:
            r = ref[i]
            ids = r["ids"].to(dev)
            full = model(ids).logits.float()
            lg = full[0, :-1]
            lp = torch.log_softmax(lg, dim=-1)
            ref_lp = r["top_lp"].to(dev)
            ref_ix = r["top_ix"].to(dev)
            # bf16's argmax is column 0 of its own topk.
            agree = (lp.argmax(dim=-1) == ref_ix[:, 0]).float().sum().item()
            nll_ref = -lp.gather(-1, ref_ix[:, :1]).sum().item()
            # KL(bf16 || q) restricted to bf16's top-128 and renormalised over it.
            p = torch.softmax(ref_lp, dim=-1)
            q_lp = lp.gather(-1, ref_ix)
            q_lp = q_lp - torch.logsumexp(q_lp, dim=-1, keepdim=True)
            p_lp = ref_lp - torch.logsumexp(ref_lp, dim=-1, keepdim=True)
            kl = (p * (p_lp - q_lp)).sum().item()
            rec = {"i": i, "agree": agree, "npos": int(lp.shape[0]),
                   "nll_ref": nll_ref, "kl": kl}
            # The same two statistics restricted to the answer position and the option
            # letters -- the only distribution the bench actually reads.
            if "ans_lp" in r:
                ref_a = r["ans_lp"].to(dev)
                alp = torch.log_softmax(full[0, -1], dim=-1)[
                    torch.tensor(letter_ids(tok), device=dev)]
                alp = alp - alp.logsumexp(0)
                rec["ans_agree"] = float(alp.argmax() == ref_a.argmax())
                rec["ans_kl"] = float((ref_a.exp() * (ref_a - alp)).sum())
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    fh.close()
    print(f"  {tag}: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
