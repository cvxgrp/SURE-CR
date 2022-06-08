import math
import time

import torch

import surecr
import surecr.prox_lib as pl
import linops as lo

# Define custom linear operator
class SelectionOperator(lo.LinearOperator):
    def __init__(self, shape, idxs):
        self._shape = shape
        self._adjoint = _AdjointSelectionOperator(idxs,
                (self._shape[1], self._shape[0]), self)
        self._idxs = idxs

    def _matmul_impl(self, X):
        return X[self._idxs]

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b):
        LHS = torch.ones_like(b)
        LHS[self._idxs] += lambda_
        return b / LHS

class _AdjointSelectionOperator(lo.LinearOperator):
    def __init__(self,  idxs, shape, adjoint):
        self._shape = shape
        self._adjoint = adjoint
        self._idxs = idxs

    def _matmul_impl(self, y):
        z = torch.zeros(self.shape[0], dtype=y.dtype, device=y.device)
        z[self._idxs] = y
        return z.reshape(-1)

n = 100
m = 2 * n
variance = 2
d = int(0.1 * m * n)
p = m * n
max_sigma_value = n
rank = max(int(0.02 * n), 5)

def run_experiment():
    revealed_indices, y, lambda_val = generate_data()
    A = SelectionOperator((d, p), revealed_indices)

    # Generates a function that applies singular value thresholding, but which
    # uses a continous extension of the derivative for the .backward method.
    prox = pl.make_scaled_prox_nuc_norm((m, n), lambda_val)

    y_cuda = y.cuda()

    hutch_seed = n * 654
    torch.manual_seed(hutch_seed)

    t0 = time.monotonic()
    solver = surecr.ADMMSolver(A, prox, torch.zeros(p).cuda(), device=y_cuda.device)
    sure = surecr.SURE(variance, solver)

    sure_val = sure.compute(y_cuda)
    tf = time.monotonic()

    print(f"SURE: {sure_val}, duration: {tf - t0}")

def generate_data():
    torch.random.manual_seed(10)
    U, _, VT = torch.linalg.svd(torch.rand((m, n)))
    Sigma = max_sigma_value * torch.rand(rank)
    beta = U[:, :rank] @ torch.diag(Sigma) @ VT[:rank, :]
    revealed_indices = torch.randperm(m * n)[:d]

    A = SelectionOperator((d, p), revealed_indices)

    lambda_max = torch.linalg.matrix_norm(
        (A.T @
            ((A @ beta.reshape(-1)) + math.sqrt(variance) * torch.randn(d))
        ).reshape((m, n)), ord=2)

    seed = n * 11
    torch.manual_seed(seed)
    y = A @ beta.reshape((-1)) + math.sqrt(variance) * torch.randn(A.shape[0])

    lambda_val = (0.25 * lambda_max).cuda()
    return revealed_indices, y, lambda_val

if __name__ == '__main__':
    run_experiment()
