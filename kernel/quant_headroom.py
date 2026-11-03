#!/usr/bin/env python3
"""How much of the activation-quantizer cost is launch overhead?

`e2e_perf.py` shows the quantizer costing ~4x the GEMM it feeds, and costing the same
at M=1 as at M=128. Flat cost across a 128x change in data volume is the signature of
launch-bound code, not bandwidth-bound code -- `prepare_a` is a few dozen small torch
ops, and at these sizes each one is dominated by its own dispatch.

Two measurements settle it:

  scaling   quantizer ms vs M. Flat => launch-bound. Linear => bandwidth-bound.
  compiled  the same function under torch.compile, which fuses the elementwise chain.

If the compiled version is much faster, the remaining gap to bf16 is an integration
artefact with a known fix, not a property of the kernel. If it is not, the quantizer
is doing real work and a fused kernel is the only way out.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/root")
sys.path.insert(0, "/work")

import torch

from kernel import qlayout as ql

ARMS = ["fp4", "w4a6", "fp6", "w4a4_tc", "w6a6_tc"]
K = 3840


def _time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    dev = torch.device("cuda")
    Ms = [1, 128, 1024, 8192]

    print(f"activation quantizer, K={K}, ms per call")
    print(f"{'arm':10s}" + "".join(f"{'M=' + str(m):>12s}" for m in Ms)
          + f"{'8192/1':>10s}")
    for arm in ARMS:
        a = ql.canon_arm(arm)
        row, first = [], None
        for M in Ms:
            x = torch.randn(M, K, device=dev)
            t = _time(lambda: ql.prepare_a(x, a), iters=10)
            row.append(t)
            first = first or t
        ratio = row[-1] / row[0] if row[0] else 0.0
        print(f"{arm:10s}" + "".join(f"{t:12.3f}" for t in row) + f"{ratio:10.1f}x")

    print("\ndata volume grows 8192x from M=1 to M=8192; a bandwidth-bound quantizer")
    print("would grow with it. A flat row is launch overhead.\n")

    print("eager vs torch.compile, M=128")
    print(f"{'arm':10s}{'eager':>10s}{'compiled':>10s}{'speedup':>10s}")
    x = torch.randn(128, K, device=dev)
    for arm in ARMS:
        a = ql.canon_arm(arm)
        eager = _time(lambda: ql.prepare_a(x, a), iters=10)
        try:
            fn = torch.compile(ql.prepare_a, dynamic=False)
            comp = _time(lambda: fn(x, a), iters=10, warmup=2)
            print(f"{arm:10s}{eager:10.3f}{comp:10.3f}{eager / comp:9.2f}x")
        except Exception as e:
            print(f"{arm:10s}{eager:10.3f}  compile failed: {type(e).__name__}: "
                  f"{str(e)[:60]}")


if __name__ == "__main__":
    main()
