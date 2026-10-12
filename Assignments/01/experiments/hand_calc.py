"""Required hand calculation (Figures 1 and 2): 3 tokens, d_k = d_v = 2. Run: python -m experiments.hand_calc"""
import torch

torch.set_printoptions(precision=4, sci_mode=False)
Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
d_k = 2

print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)
S = Q @ K.T
print("\n1) QK^T =\n", S)
S_scaled = S / d_k ** 0.5
print("\n2) / sqrt(d_k) =\n", S_scaled)
W = torch.softmax(S_scaled, dim=-1)
print("\n3) row-wise softmax =\n", W)
print("\n4) output = W @ V =\n", W @ V)

mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
S_masked = S_scaled.masked_fill(~mask, float("-inf"))
print("\n5) causal-masked scores =\n", S_masked)
Wc = torch.softmax(S_masked, dim=-1)
print("\n6) causal weights =\n", Wc)
print("\n7) causal output =\n", Wc @ V)
print("\nToken 1 only sees itself (output = V[0]); token 2 sees 1-2; token 3 sees 1-3.")
