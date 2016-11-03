"""Inference algorithms for the Plackett--Luce model."""

from __future__ import division

import math
import numpy as np
import random
import scipy.linalg as spl
import warnings


def lsr(nb_items, rankings, initial_strengths=None):
    """Computes a fast estimate of Plackett--Luce model parameters.
    
    Items are expected to be represented by consecutive integers from `0` to
    `n-1`. A (partial) ranking is defined by a tuple containing the items in
    decreasing order of preference. For example, the tuple
    
        (2, 0, 4)

    corresponds to a ranking where `2` is first, `0` is second, and `4` is
    third.

    The estimate is found using the Luce Spectral Ranking algorithm (LSR).

    The argument `initial_strengths` can be used to iteratively refine an
    existing parameter estimate (see the implementation of `ilsr` for an idea
    on how this works).

    Args:
        nb_items (int): The number of distinct items.
        rankings (List[tuple]): The data (partial rankings).
        initial_strengths (Optional[List]): Strengths used to parametrize the
            transition rates of the LSR Markov chain. If `None`, the strengths
            are assumed to be uniform over the items.

    Returns:
        strengths (List[float]): an estimate of the model parameters given
            the data.
    """
    if initial_strengths is None:
        ws = np.ones(nb_items)
    else:
        ws = np.asarray(initial_strengths)
    chain = np.zeros((nb_items, nb_items), dtype=float)
    for ranking in rankings:
        sum_ = sum(ws[x] for x in ranking)
        for i, winner in enumerate(ranking[:-1]):
            val = 1.0 / sum_
            for loser in ranking[i+1:]:
                chain[loser, winner] += val
            sum_ -= ws[winner]
    chain -= np.diag(chain.sum(axis=1))
    try:
        return statdist(chain)
    except:
        # Ideally we would like to catch `spl.LinAlgError` only, but there seems
        # to be a bug in scipy, in the code that raises the LinAlgError (!!).
        raise ValueError("the comparison graph is not strongly connected")


def ilsr(nb_items, rankings, max_iter=100, eps=1e-8):
    """Compute the ML estimate of Plackett--Luce model parameters.
    
    Items are expected to be represented by consecutive integers from `0` to
    `n-1`. A (partial) ranking is defined by a tuple containing the items in
    decreasing order of preference. For example, the tuple
    
        (2, 0, 4)

    corresponds to a ranking where `2` is first, `0` is second, and `4` is
    third.

    The estimate is found using the Iterative Luce Spectral Ranking algorithm
    (I-LSR).

    Args:
        nb_items (int): The number of distinct items.
        rankings (List[tuple]): The data (partial rankings.)
        max_iter (Optional[int]): The maximum number of iterations.
        eps (Optional[float]): Minimum difference between successive
            log-likelihoods to declare convergence.

    Returns:
        strengths (List[float]): the ML estimate of the model parameters given
            the data.

    Raises:
        RuntimeError: If the algorithm does not converge after `max_iter`
            iterations.
    """
    strengths = np.ones(nb_items)
    prev_loglik = -np.inf
    for _ in range(max_iter):
        strengths = lsr(nb_items, rankings, initial_strengths=strengths)
        loglik = log_likelihood(rankings, strengths)
        if abs(loglik - prev_loglik) < eps:
            return strengths
        prev_loglik = loglik
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))


def log_likelihood(rankings, strengths):
    """Compute the log-likelihood of Plackett--Luce model parameters.

    Args:
        rankings (List[tuple]): The data (partial rankings.)
        strengths (List[float]): The model parameters.

    Returns:
        loglik (float): the log-likelihood of the parameters given the data.
    """
    loglik = 0
    for ranking in rankings:
        sum_ = sum(strengths[x] for x in ranking)
        for i, winner in enumerate(ranking[:-1]):
            loglik += math.log(strengths[winner])
            loglik -= math.log(sum_)
            sum_ -= strengths[winner]
    return loglik


def statdist(generator):
    """Compute the stationary distribution of a Markov chain.

    Args:
        generator (numpy.ndarray): The infinitesimal generator matrix of the
        Markov chain.

    Returns:
        dist (List[float]): The unnormalized stationary distribution of the
            Markov chain.
    """
    n = generator.shape[0]
    with warnings.catch_warnings():
        # The LU decomposition raises a warning when the generator matrix is
        # singular (which it, by construction, is!).
        warnings.filterwarnings('ignore')
        lu, piv = spl.lu_factor(generator.T, check_finite=False)
    # The last row contains 0's only.
    left = lu[:-1,:-1]
    right = -lu[:-1,-1]
    # Solves system `left * x = right`. Assumes that `left` is
    # upper-triangular (ignores lower triangle).
    res = spl.solve_triangular(left, right, check_finite=False)
    res = np.append(res, 1.0)
    return (n / res.sum()) * res


def generate_rankings(strengths, nb_rankings, size_of_ranking=3):
    """Generate random rankings according to a Plackett--Luce model.

    Args:
        strengths (List[float]): The model parameters.
        nb_rankings (int): The number of rankings to generate.
        size_of_ranking (Optional[int]): The number of items to include in each
            ranking. Default value: 3.

    Returns:
        data (List[tuple]): A list of (partial) rankings generated according to
            a Plackett--Luce model with the specified model parameters.
    """
    n = len(strengths)
    items = range(n)
    data = list()
    for _ in range(nb_rankings):
        alts = random.sample(items, size_of_ranking)
        probs = np.array([strengths[x] for x in alts])
        datum = list()
        for _ in range(size_of_ranking):
            probs /= np.sum(probs)
            idx = np.random.choice(size_of_ranking, p=probs)
            datum.append(alts[idx])
            probs[idx] = 0.0
        data.append(tuple(datum))
    return tuple(data)
