"""Required hand calculation (report Figure 1 and 2).

Tokens: "I love AI", d_k = d_v = 2. The Q/K/V vectors are hypothetical, already-projected values
chosen so the arithmetic is easy to do on paper.
Run: python -m experiments.hand_calc
"""
import os
import torch

torch.set_printoptions(precision=4, sci_mode=False)
tokens = ["I", "love", "AI"]
Q = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 0.], [0., 2.], [2., 2.]])
d_k = 2


def run(verbose=True):
    r = {}
    r["S"] = Q @ K.T
    r["S_scaled"] = r["S"] / d_k ** 0.5
    r["W"] = torch.softmax(r["S_scaled"], dim=-1)
    r["out"] = r["W"] @ V
    mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    r["mask"] = mask
    r["S_masked"] = r["S_scaled"].masked_fill(~mask, float("-inf"))
    r["Wc"] = torch.softmax(r["S_masked"], dim=-1)
    r["out_c"] = r["Wc"] @ V
    if verbose:
        print("tokens:", tokens)
        print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)
        print("\n1) QK^T =\n", r["S"])
        print("\n2) QK^T / sqrt(d_k) =\n", r["S_scaled"])
        print("\n3) row-wise softmax weights =\n", r["W"])
        print("\n4) output = weights @ V =\n", r["out"])
        print("\n5) causal-masked scores =\n", r["S_masked"])
        print("\n6) causal weights =\n", r["Wc"])
        print("\n7) causal output =\n", r["out_c"])
        print("\n'I' sees only itself; 'love' sees I+love; 'AI' sees all three.")
    return r


if __name__ == "__main__":
    run()
