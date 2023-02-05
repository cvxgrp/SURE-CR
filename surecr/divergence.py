from __future__ import annotations

import torch
import math
from typing import Callable 

import linops as lo
import linops.trace as lot

"""
This module contains various ways to compute divergences.

Its API is unstable and internal to sure.


Reference paper: https://arxiv.org/pdf/2010.09649.pdf
"""

def divergence(solution, input_vec, strategy, parameters) -> torch.Tensor:
    J = lo.VectorJacobianOperator(solution, input_vec)
    
    if strategy == 'exact':
        return lot.exact_trace(J, **parameters)
    elif strategy == 'hutchinson':
        return lot.hutchinson(J, **parameters)
    elif strategy == 'hutch++':
        return lot.hutchpp(J, **parameters)
    elif strategy == 'xtrace':
        return lot.xtrace(J, **parameters)[0]
    elif strategy == 'default' or strategy == 'xnystrace':
        return lot.xnystrace(J, **parameters)[0]
    else:
        raise RuntimeError("Unknown divergence strategy.")
