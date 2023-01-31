from __future__ import annotations

import torch
import math
from typing import Callable 

"""
This module contains various ways to compute divergences.

Its API is unstable and internal to sure.


Reference paper: https://arxiv.org/pdf/2010.09649.pdf
"""

def vjp_oracle_factory(solution, input_vec):
    def oracle(z, retain_graph=True):
        solution.backward(gradient=z, retain_graph=retain_graph)
        grad = input_vec.grad
        input_vec.grad = None
        return grad
    return oracle

def hutchinson(solution: torch.Tensor, input_vec: torch.Tensor, m: int=1000):
    if solution.numel() <= m:
        return exact_divergence(solution, input_vec)
    vjp = vjp_oracle_factory(solution, input_vec)
    total = 0
    for _ in range(m):
        z = (2 * torch.randint(2, size=solution.shape, device=solution.device) - 1).float()
        total = total + (z * vjp(z)).sum()
    return total / m



def hutch_pp(solution: torch.Tensor, input_vec: torch.Tensor, m: int=102):
    """
    Algorithm taken from reference paper above on Hutch++.
    """
    assert m % 3 == 0
    if solution.numel() <= m:
        return exact_divergence(solution, input_vec)
    vjp = vjp_oracle_factory(solution, input_vec)
    k = m // 3
    S = 2.0 * torch.randint(0, 2, (*solution.shape, k), device=input_vec.device) - 1.0
    G = 2.0 * torch.randint(0, 2, (*solution.shape, k), device=input_vec.device) - 1.0
    BS = torch.empty_like(S)
    for i in range(k):
        BS[:, i] = vjp(S[:,i])
 
    Q, _ = torch.linalg.qr(BS)
    G_prime = G - Q @ (Q.T @ G)
    total = 0
    for i in range(Q.shape[1]):
        z = Q[:,i]
        total = total + (z * vjp(z)).sum()

    for i in range(k - 1):
        z = G_prime[:,i]
        total = total + (z * vjp(z)).sum() / k
    z = G_prime[:,-1]
    total = total + (z * vjp(z, False)).sum() / k

    return total


def xtrace(solution: torch.Tensor, input_vec: torch.Tensor, m: int=60):
    n = solution.numel()
    if n <= m:
        return exact_divergence(solution, input_vec)
    vjp = vjp_oracle_factory(solution, input_vec)

    m = m // 2
    def normalize_columns(M):
        return M / torch.linalg.vector_norm(M, dim=0)

    def diag_of_AB(A, B):
        return torch.sum(A * B, dim=0)

    Z = torch.randn(n, m, device=input_vec.device)
    Omega = math.sqrt(n) * normalize_columns(
        torch.randn(n, m, device=input_vec.device)
    )
    Y = torch.empty_like(Omega)
    for i in range(m):
        Y[:, i] = vjp(Omega[:,i])
    Q, R = torch.linalg.qr(Y)

    W = Q.T @ Omega
    S = normalize_columns(torch.linalg.inv(R).T)
    scale = (n - m + 1) / (n - torch.linalg.vector_norm(W, dim=0)**2 +
                           torch.abs(diag_of_AB(S, W) * torch.linalg.vector_norm(S, dim=0))**2)

    Z = torch.empty_like(Q)
    for i in range(m):
        Z[:, i] = vjp(Q[:,i])

    H = Q.T @ Z
    HW = H @ W
    T = Z.T @ Omega
    dSW = diag_of_AB(S, W)
    dSHS = diag_of_AB(S, H @ S)
    dTW = diag_of_AB(T, W)
    dWHW = diag_of_AB(W, HW)
    dSRmHW = diag_of_AB(S, R - HW)
    dTmHRS = diag_of_AB(T - H.T @ W, S)

    ests = torch.trace(H) - dSHS + (
            dWHW - dTW + dTmHRS * dSW + torch.abs(dSW)**2 * dSHS + dSW * dSRmHW) * scale

    return torch.mean(ests), torch.std(ests) / math.sqrt(m)


def exact_divergence(solution: torch.Tensor, input_vec: torch.Tensor):
    divergence = 0
    for (i, gi) in enumerate(solution):
        gi.backward(retain_graph=True)
        divergence = divergence + input_vec.grad[i]
        input_vec.grad = None
    return divergence
