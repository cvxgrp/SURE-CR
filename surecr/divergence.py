from __future__ import annotations

import torch
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

def exact_divergence(solution: torch.Tensor, input_vec: torch.Tensor):
    divergence = 0
    for (i, gi) in enumerate(solution):
        gi.backward(retain_graph=True)
        divergence = divergence + input_vec.grad[i]
        input_vec.grad = None
    return divergence
