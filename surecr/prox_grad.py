import torch
import math
from typing import Callable

# We can use the actual lipschitz constant with respect with just the max
# singular value  of A.T A and remove the line search.

def _line_search(x, lambda_init, f, grad_f, prox_g):
    """ https://stanford.edu/~boyd/papers/pdf/prox_slides.pdf; Slide 20 """
    beta = 0.5
    lambda_ = lambda_init
    grad_fx = grad_f(x)
    f_x = f(x)
    while True:
        z = prox_g(x - lambda_ * grad_fx, lambda_)
        norm_z_m_x = torch.linalg.vector_norm(z - x)
        if f(z) <= f_x + grad_fx @ (z - x) + 1/2 / lambda_ * norm_z_m_x**2:
            break
        lambda_ = beta * lambda_

    return lambda_, z, norm_z_m_x

def prox_grad_w_linesearch(f: Callable[[torch.Tensor], torch.Tensor],
              grad_f: Callable[[torch.Tensor], torch.Tensor],
              prox_g: Callable[[torch.Tensor], torch.Tensor],
              x0: torch.Tensor, _, max_iters=5000, eps=1e-3) -> tuple[torch.Tensor, int]:
    x = x0
    norm_delta_x = float('inf')
    lambda_ = 1.0
    k = 0
    epsilon = eps * math.sqrt(x0.numel())

    while norm_delta_x > epsilon and k < max_iters:
        lambda_, x, norm_delta_x = _line_search(x, lambda_, f, grad_f, prox_g)
        k += 1

    return x, k

def accel_prox_grad_w_linesearch(f: Callable[[torch.Tensor], torch.Tensor],
                    grad_f: Callable[[torch.Tensor], torch.Tensor],
                    prox_g: Callable[[torch.Tensor], torch.Tensor],
                    x0: torch.Tensor, _, max_iters=5000, eps=1e-3) -> tuple[torch.Tensor, int]:
    xprev = 0
    x = x0
    norm_delta_x = float('inf')
    lambda_ = 1.0
    k = 0
    epsilon = eps * math.sqrt(x0.numel())

    while norm_delta_x > epsilon and k < max_iters:
        y = x + k / (k + 3) * (x - xprev)
        xprev = x
        lambda_, x, _ = _line_search(y, lambda_, f, grad_f, prox_g)
        norm_delta_x = torch.linalg.vector_norm(x - xprev).item()
        k += 1

    return x, k

def prox_grad(_,
              grad_f: Callable[[torch.Tensor], torch.Tensor],
              prox_g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              x0: torch.Tensor,
              lipschitz_constant: torch.Tensor,
              max_iters=5000, eps=1e-3) -> tuple[torch.Tensor, int]:
    x = x0
    lambda_ = 0.8 / lipschitz_constant
    k = 0
    grad_fx = grad_f(x)
    err = float('inf')
    epsilon = eps * math.sqrt(x0.numel())

    while err > epsilon and k < max_iters:
        #lambda_, x, norm_delta_x = _line_search(x, lambda_, f, grad_f, prox_g)
        intermediate_x = x - lambda_ * grad_fx
        x = prox_g(intermediate_x, lambda_)
        k += 1
        grad_fx = grad_f(x)
        err = torch.linalg.vector_norm(
                grad_fx - 1 / lambda_ * (x - intermediate_x)
            )
        #print(f"{k=}, {err=}, {lambda_=}")

    return x, k

def accel_prox_grad(_,
                    grad_f: Callable[[torch.Tensor], torch.Tensor],
                    prox_g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    x0: torch.Tensor,
                    lipschitz_constant: torch.Tensor,
                    max_iters=5000, eps=1e-3) -> tuple[torch.Tensor, int]:
    xprev = 0
    x = x0
    err = float('inf')
    lambda_ = 0.8 / lipschitz_constant
    k = 0
    epsilon = eps * math.sqrt(x0.numel())

    while err > epsilon and k < max_iters:
        y = x + k / (k + 3) * (x - xprev)
        xprev = x

        intermediate_y = y - lambda_ * grad_f(y)
        x = prox_g(intermediate_y, lambda_)
        k += 1
        err = torch.linalg.vector_norm(
                grad_f(x) - 1 / lambda_ * (x - intermediate_y)
            )
        #print(f"{k=}, {err=}, {lambda_=}")

    return x, k
