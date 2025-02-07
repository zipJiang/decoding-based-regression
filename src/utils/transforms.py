""" """

import numpy as np


_BETA_POSITIVE_ = .02
_BETA_NEGATIVE_ = .3


def _inverse_beta_sigmoid(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    beta_sigmoid(x) = 1 / (1 + exp(-beta * x))
    inverse_beta_sigmoid(x) = ln(1 / x - 1) / -beta
    """
    
    return np.log(1 / x - 1) / (-beta)  # Make sure that your data is broadcastable


def _beta_sigmoid(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    beta_sigmoid(x) = 1 / (1 + exp(-beta * x))
    """
    
    return 1 / (1 + np.exp(-beta * x))  # Make sure that your data is broadcastable


_MIN_UNLI_ = _beta_sigmoid(-50, _BETA_NEGATIVE_)
_MAX_UNLI_ = _beta_sigmoid(50, _BETA_POSITIVE_)


def _inverse_sigmoid_unli(x: np.ndarray) -> np.ndarray:
    """ """

    sigmoid_inversed = np.where(
        x <= .5,
        _inverse_beta_sigmoid(x * (1 - 2 * _MIN_UNLI_) + _MIN_UNLI_, _BETA_NEGATIVE_),
        _inverse_beta_sigmoid(.5 + (x - .5) * (2 * _MAX_UNLI_ - 1), _BETA_POSITIVE_)
    )
    
    return (sigmoid_inversed + 50) * 100