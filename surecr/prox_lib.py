import torch
from typing import Callable, Union
import sys

if sys.version_info >= (3, 10):
    floatTensor = float | torch.Tensor
else:
    floatTensor = Union[float,  torch.Tensor]
    

ProxOperator = Callable[[torch.Tensor, floatTensor], torch.Tensor]

def prox_l1_norm(v: torch.Tensor, t: floatTensor) -> torch.Tensor:
    """
    The ell_1 norm's proximal operator.
    """
    return torch.sign(v) * torch.relu(torch.abs(v) - t)

def make_scaled_prox_nuc_norm(shape: tuple[int, int], t_scale: floatTensor) -> \
        ProxOperator:
    """
    generates the proximal operator prox_r of r(b) = t_scale ||b||_*
    """
    t_scale = torch.as_tensor(t_scale)
    m, n = shape
    if m >= n:
        def prox_nuc_norm(x, t):
            X = x.reshape(shape)
            return _prox_nuc_norm.apply(X, t_scale * t).reshape(-1)
        return prox_nuc_norm
    else:
        def prox_nuc_norm_T(x, t):
            X = x.reshape(shape)
            return _prox_nuc_norm.apply(X.T, t_scale * t).T.reshape(-1)
        return prox_nuc_norm_T

def combine_proxs(shapes: list[int], proxs: list[ProxOperator]) -> ProxOperator:
    """
    Combines lists of proximal operators for seperable regularizers.

    If prox_r_1: R^a -> R^a and prox_r_2: R^b -> R^b then the arguments to this
    function should be [a, b], [prox_r_1, prox_r_2].
    """
    def combined_prox(v, t):
        total = 0
        out = torch.empty_like(v)
        for shape, prox in zip(shapes, proxs):
            mask = slice(total, total + shape)
            out[mask] = prox(v[mask], t)
            total += shape
        return out
    return combined_prox

def scale_prox(prox: ProxOperator, t_scale: floatTensor) -> ProxOperator:
    """
    Takes a proximal operator of r, and returns the proximal operator of t_scale r.
    """
    return lambda v, t: prox(v, t_scale * t)


def prox_l2_norm(v, t):
    """
    The ell_2 norm's proximal operator.
    """
    return torch.relu(1 - t / torch.linalg.norm(v)) * v

class _prox_nuc_norm(torch.autograd.Function):
    """
        Let X.shape == (m, n)
        This assumes m >= n.
    """
    @staticmethod
    #@torch.jit.script
    def forward(ctx, X, lambda_):
        def f(s):
            return torch.relu(s - lambda_)

        U, S, VT = torch.linalg.svd(X, full_matrices=True)
        ctx.U = U
        ctx.S = S
        ctx.VT = VT
        ctx.save_for_backward(lambda_)
        return U[:, :X.shape[1]] @ torch.diag(f(S)) @ VT

    @staticmethod
    #@torch.jit.script
    def backward(ctx, grad_output):
        epsilon = 1e-3
        lambda_, = ctx.saved_tensors
        U, S, VT = ctx.U, ctx.S, ctx.VT
        Z = grad_output
        m, n = Z.shape
        assert m >= n

        def f(s):
            return torch.relu(s - lambda_)
        def fp(s):
            #return (s >= lambda_).double()
            return (s >= lambda_).float()

        zeta = U.T @ Z @ VT.T
        Gamma = torch.empty_like(zeta)

        if m > n:
            R_mask = (S >= epsilon)
            R = torch.empty_like(S)
            R[R_mask] = f(S[R_mask]) / S[R_mask]
            R[~R_mask] = fp(S[~R_mask])
            Gamma[n:, :] = torch.tile(R, (m - n, 1)) * zeta[n:, :]

        S_i = torch.tile(S, (n, 1)).T 
        S_j = torch.tile(S, (n, 1))
        def off_diags_zeta(S_i, S_j):
            return (S_i * f(S_i) - S_j * f(S_j)) / (S_i**2 - S_j**2)
        def off_diags_zetaT(S_i, S_j):
            return (S_j * f(S_i) - S_i * f(S_j)) / (S_i**2 - S_j**2)
        zeta_weights = torch.empty_like(zeta[:n,:])
        zetaT_weights = torch.empty_like(zeta[:n,:])

        mask = torch.abs(S_i - S_j) <= epsilon

        zeta_weights[~mask] = off_diags_zeta(S_i[~mask], S_j[~mask])
        zetaT_weights[~mask] = off_diags_zetaT(S_i[~mask], S_j[~mask])

        def off_diags_pos_repeated_zeta(S):
            return (1/2 * fp(S) + 1/2 * f(S) / S)

        def off_diags_pos_repeated_zetaT(S):
            return (1/2 * fp(S) - 1/2 * f(S) / S)
        pos_mask = (mask & (S_i >= epsilon))
        zeta_weights[pos_mask] = off_diags_pos_repeated_zeta(S_i[pos_mask])
        zetaT_weights[pos_mask] = off_diags_pos_repeated_zetaT(S_i[pos_mask])

        zero_mask = (mask & (S_i < epsilon))
        zeta_weights[zero_mask] = fp(S_i[zero_mask])
        zetaT_weights[zero_mask] = 0

        torch.diagonal(zeta_weights)[:] = fp(S)
        torch.diagonal(zetaT_weights)[:] = 0

        Gamma[:n, :] = (
            zeta_weights * zeta[:n,:] + zetaT_weights * zeta[:n,:].T
        )

        retval = U @ Gamma @ VT
        return retval, None, None
