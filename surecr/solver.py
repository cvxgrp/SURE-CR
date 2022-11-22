import torch
import surecr.admm as admm
import surecr.prox_grad as prox_grad

class Solver:
    """
    For any given instance s of a solver class, s.estimate(s.solve(y)) must be
    differentiable via torch's backpropagation.
    """
    def __init__(self):
        raise NotImplementedError()
    def solve(self, y):
        """
        Returns intermediate value used to estimate the mean of the distribution
        y is sampled from.
        """
        raise NotImplementedError()
    def estimate(self, beta):
        """
        Given the output of a solve call, returns the estimate of the mean of the
        distribution y was sampled from.
        """
        return NotImplementedError()

class CVXPYSolver(Solver):
    def __init__(self, problem, y_parameter, variables, estimate):
        """
        problem must be a CVXPY problem with a single paremeter, y_parameter,
            and variables y_variable.
    
        estimate must be function which takes tensors with values for each variable
            and returns the estimate.

        WARNING: This solver has poor performance on large problems, and can
        have undetected poor accuracy on some moderately-sized problems.
        """
        try:
            from cvxpylayers.torch import CvxpyLayer
        except ImportError:
            raise RuntimeError("Could not import cvxpylayers. Please ensure it is"
                               " correctly installed. Note: cvxpylayers is not"
                               "automatically installed when using conda.")
        self._problem = problem
        self._y_parameter = y_parameter
        self._variables = variables
        self._layer = CvxpyLayer(problem, parameters=[y_parameter], variables=variables)
        self._estimate = estimate

    def solve(self, y):
        return self._layer.forward(y)[0]
    
    def estimate(self, beta):
        return self._estimate(beta)

class ADMMSolver(Solver):
    """
    This solver solves problems of the form with a variant on ADMM:
            min. 1/2 ||A b - y||_2^2 + r(b)
    and estimates the mean of y with A b^* where b^* is the optimal b.

    A is a linear operator defined using <https://github.com/cvxgrp/torch_linops>

    prox_R is a differentiable-with-respect-to-its-first-argument function to
        find the optimal point b for a (v, t) pair of
            min. t r(b) + 1/2 ||b - v||_2^2

    x0 is the point where we begin iterations, it must be chosen
        indepentently of y.

    max_iters, eps_rel, eps_abs control when iterations stop.
    """
    def __init__(self, A, prox_R, x0, device=None, **parameters):
        self._A = A
        self._prox_R = prox_R
        self._x0 = x0
        self._device = device
        self._iters = None
        self._parameters = parameters

    def solve(self, y):
        x, z, _, self._iters = admm.admm(A=self._A, b=y,
                                         prox_R=self._prox_R,
                                         x0=self._x0,
                                         device=self._device,
                                         **self._parameters)
        return (x + z) / 2
    
    def estimate(self, parameters):
        return self._A @ parameters

class ISTASolver(Solver):
    _prox_grad_solver = lambda _, *args, **kwargs: prox_grad.prox_grad(*args, **kwargs)

    def __init__(self, A, prox_R, x0, device=None,
                 lipschitz_iterations=20,
                 lipschitz_vec=None, **parameters):
        self._A = A
        self._prox_R = prox_R
        self._x0 = x0
        self._device = device
        self._iters = None
        self._lipschitz = _compute_lipschitz(
                A,
                torch.ones_like(x0.reshape(-1)) / x0.numel() \
                        if lipschitz_vec is None else lipschitz_vec,
                lipschitz_iterations)
        self._parameters = parameters

    def solve(self, y):
        self._y = y
        self._ATb = self._A.T @ y
        x, self._iters = self._prox_grad_solver(self._f,
                                                self._grad_f,
                                                self._prox_R,
                                                self._x0.reshape(-1),
                                                self._lipschitz,
                                                **self._parameters)
        return x.reshape(self._x0.shape)

    def estimate(self, beta):
        return self._A @ beta

    def _f(self, x):
        return 1/2 * torch.linalg.vector_norm(self._A @ x - self._y)**2

    def _grad_f(self, x):
        """ f(x) = 12 ||Ax -  b||_2^2 = 1/2 (A x - b)^T (Ax - b) =
            1/2 (x^T A^T A x - b^T Ax - x^T A^T b + ||b||_2^2) = 
            1/2 x^T A^T A x - x^T A^T b + 1/2 ||b||_2^2

            grad f(x) = A^T A x -  A^T b
        """
        return self._A.T @ (self._A @ x) - self._ATb

class FISTASolver(ISTASolver):
    """
    This solver solves problems of the form with a variant on FISTA:
            min. 1/2 ||A b - y||_2^2 + r(b)
    and estimates the mean of y with A b^* where b^* is the optimal b.

    A is a linear operator defined using <https://github.com/cvxgrp/torch_linops>

    prox_R is a differentiable-with-respect-to-its-first-argument function to
        find the optimal point b for a (v, t) pair of
            min. t r(b) + 1/2 ||b - v||_2^2

    x0 is the point where we begin iterations, it must be chosen
        indepentently of y.

    lipschitz_iterations is how many iterations of the power method to use
    to approximate the largest eigenvalue of A^T A

    lipschitz_vec is the vector to start the power method. By default, a
    vector of all 1s is used. If this vector is orthogonal to the largest
    eigenvector of A^T A, this argument is mandatory.

    max_iters, eps control when iterations stop.
    """

    #_prox_grad_solver = lambda _, *args: prox_grad.accel_prox_grad_w_linesearch(*args)
    _prox_grad_solver = lambda _, *args, **kwargs: prox_grad.accel_prox_grad(*args, **kwargs)

def _compute_lipschitz(A, x, iters):
    """
    Finds Lipschitz constant of _grad f(x) = A^T A x -  A^T b

    ||A^T A x - A^Tb - A^T A y + A^T b||_2 =
        ||A^T A x - A^T A y||_2 <= ||A^T A|| ||x - y||_2
    Where ||A^T A|| = sigma_1(A^T A), the maximal singular value of A^T A
    """
    assert torch.linalg.vector_norm(x) > 0
    norm_x = float('inf')

    for _ in range(iters):
        x = A.T @ (A @ x)
        norm_x = torch.linalg.vector_norm(x)
        x /= norm_x
        if torch.abs(norm_x).item() <= 1e-6:
            raise RuntimeError("Either uniform vector or provided "
                               "lipschitz vector is in A^T A's null space. "
                               "Please provide a vector which is NOT "
                               "orthogonal to A^T A's largest eigenvector.")
    #print(f"Lipschitz constant for A: {norm_x}")

    return norm_x
