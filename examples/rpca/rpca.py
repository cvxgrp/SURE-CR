import math
import time

import torch
import surecr
import surecr.prox_lib as pl
import linops as lo

n = 100
m = n

d = m * n
p = 2 * d

max_sigma_value = n
rank = max(5, int(0.02 * n))
non_zeros = max(10, int(1e-4 * n**2))
max_S_entry = 100
variance = 2

def run_experiment():
    y, lambda_val, gamma_val = generate_data()
    A = lo.hstack([lo.IdentityOperator(m * n), lo.IdentityOperator(m * n)])
    y_cuda = y.cuda()

    # Generates a function that applies singular value thresholding, which uses a
    # continous extension of the derivative for the .backward method.
    prox_L = pl.make_scaled_prox_nuc_norm((m, n), lambda_val)

    def prox_S(v, t):
        return torch.relu(v - gamma_val * t) - torch.relu(-v - gamma_val * t)

    def prox(v, t):
        return torch.hstack([
            prox_L(v[:d], t), prox_S(v[d:], t)
        ])
    # Alternatively,
    #prox = pl.combine_proxs([d, d], [prox_L, prox_S])

    hutch_seed = m * 326

    torch.manual_seed(hutch_seed)

    solver = surecr.ADMMSolver(A, prox, torch.zeros(p).cuda(), device=y_cuda.device)

    t0 = time.monotonic()
    sure = surecr.SURE(variance, solver)
    sure_val = sure.compute(y_cuda)
    tf = time.monotonic()

    print(f"SURE: {sure_val}, duration: {tf - t0}")

def generate_data():
    torch.random.manual_seed(11)
    U, _, VT = torch.linalg.svd(torch.rand((m, n)))
    Sigma = max_sigma_value * torch.rand(rank)
    L = U[:, :rank] @ torch.diag(Sigma) @ VT[:rank, :]
    flat_indices = torch.randperm(m * n)[:non_zeros]
    S = torch.zeros(m * n)
    S[flat_indices] = max_S_entry * torch.rand(non_zeros)
    beta = torch.hstack([L.reshape(-1), S])

    A = lo.hstack([lo.IdentityOperator(m * n), lo.IdentityOperator(m * n)])

    y_for_max = (A @ beta + math.sqrt(variance) * torch.randn(m * n)).reshape((m, n))
    lambda_max = torch.linalg.matrix_norm(y_for_max, ord=2)
    gamma_max = torch.abs(y_for_max).max()

    lambda_val = 0.16 * lambda_max
    gamma_val = 0.057 * gamma_max

    seed = m * 104
    torch.random.manual_seed(seed)
    y = (A @ beta.reshape((-1,))) + math.sqrt(variance) * torch.randn(A.shape[0])
    return y, lambda_val, gamma_val

if __name__ == '__main__':
    run_experiment()
