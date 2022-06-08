import math
import time

import torch

import cvxpy as cp

import surecr
import linops as lo


# Parameters
d = 250
p = 2 * d
seed = d + 7
r2 = 0.8
lambda_over_lambda_max = 0.1
variance = 2.

def run_experiment():
    X, beta, y, lambda_val = generate_data()

    beta_cvx = cp.Variable(p)
    y_cvx = cp.Parameter(d)
    prob = cp.Problem(
        cp.Minimize(
            1/2 * cp.sum_squares(
                X @ beta_cvx - y_cvx
            ) + lambda_val * cp.pnorm(beta_cvx, 1)
    ))

    solver = surecr.CVXPYSolver(prob, y_cvx, [beta_cvx], lambda b: X @ b)

    torch.manual_seed(d + 76)

    t0 = time.monotonic()

    sure = surecr.SURE(variance, solver)
    sure_val = sure.compute(y)
    tf = time.monotonic()
    print(f"SURE: {sure_val} time: {tf - t0}")

def generate_data():
    # Generate data
    torch.manual_seed(seed)

    beta_p = torch.zeros((p))
    beta_p[:d//20] = 1.
    X = torch.randn((d, p))
    mu_p_norm = torch.linalg.vector_norm(X @ beta_p)
    beta_p_scale = torch.sqrt((1 - r2) * mu_p_norm**2 / d / variance)
    beta = beta_p / beta_p_scale
    y = X @ beta + math.sqrt(variance) * torch.randn(d)
    y_for_lambda = X @ beta + math.sqrt(variance) * torch.randn(d)
    lambda_max = torch.linalg.vector_norm(X.T @ y_for_lambda, torch.inf)

    lambda_val = (lambda_over_lambda_max * lambda_max)
    return X, beta, y, lambda_val
 
if __name__ == '__main__':
    run_experiment()
