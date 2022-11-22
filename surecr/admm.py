import torch
import math

from linops import aslinearoperator


def admm(A, b, prox_R, x0, device, debug=False, max_iters=1000, eps_abs=1e-6,
        eps_rel=1e-3):
    """
        Solves problem of the form
            argmin_x R(x) + (1/2)||b - A x||_2^2

        prox_R is the proximal operator of R.
    """
    A = aslinearoperator(A)
    ATb = A.T @ b

    def prox_half_l2_norm_squared(v, lambda_):
        """
        (1/2)||A x - b||_2^2 = (1/2)(A x - b)^T (A x - b) =
            (Ax)^T Ax - (Ax)^T b - b^T Ax + b^T b =
            (1/2) x^T (A^T A) x - b^T A x + (1/2)||b||_2^2
    
        See \\S 6.1.1 of prox_algs.pdf to see that 
        prox_{lambda_ f}(v) = (I + lambda_ A^T A)^-1 (v + lambda_ A^T b)
        """
        RHS = v + lambda_ * ATb
        return A.solve_I_p_lambda_AT_A_x_eq_b(lambda_, RHS)

    problem_size = x0.shape.numel()
    rho = problem_size**1.5
    x = x0
    z = x0
    u = torch.zeros_like(x0, device=device)
    iters = 0
    if debug:
        err_pri = []
        err_dual = []
    r_pri_norm = float('inf')
    r_dual_norm = float('inf')
    x_norm = torch.linalg.vector_norm(x).item()
    z_norm = torch.linalg.vector_norm(z).item()
    u_norm = 0.0

    scaled_eps_abs =  eps_abs * math.sqrt(problem_size)
    mu = 10

    # See \S4.4 of prox_algs.pdf
    while (iters < max_iters and
            (r_pri_norm > scaled_eps_abs + eps_rel * max(x_norm, z_norm)
                or r_dual_norm > scaled_eps_abs + eps_rel * rho * u_norm)):
        if debug:
            print(f"{iters=} {x_norm=} {z_norm=} {u_norm=}")
        x = prox_R(z - u, 1/rho)
        x_norm = torch.linalg.vector_norm(x).item()
        z_new = prox_half_l2_norm_squared(x + u, 1/rho)
        r_dual_norm = rho * torch.linalg.vector_norm(z_new - z).item()
        z = z_new
        z_norm = torch.linalg.vector_norm(z).item()
        u = u + x - z
        u_norm = torch.linalg.vector_norm(u).item()
        iters += 1
        r_pri_norm = torch.linalg.vector_norm(x - z).item()

        if r_pri_norm > mu * r_dual_norm:
            if debug:
                print(f"{iters=} upping rho, {r_pri_norm=}, {r_dual_norm=}, {rho=}")
            rho = rho * 2
            u = u / 2
        elif r_dual_norm > mu * r_pri_norm:
            if debug:
                print(f"{iters=} downing rho, {r_pri_norm=}, {r_dual_norm=}, {rho=}")
            rho = rho / 2
            u = u * 2
    if debug:
        print(f"Final ADMM {iters=}")
    return x, z, u, iters
