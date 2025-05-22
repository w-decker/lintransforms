import numpy as np

def correlate(x1:np.ndarray, x2:np.ndarray, r:float, axis:int=0):
    """
    Correlate two random variables with a given correlation coefficient.

    Parameters
    ----------
    x1 : np.ndarray
        First random variable.
    x2 : np.ndarray
        Second random variable.
    r : float
        Correlation coefficient between the two random variables.
    axis : int, optional
        Axis along which the correlation is applied. Default is 0.

    Returns
    -------
    np.ndarray
        Correlated random variable.
    """
    x1 = np.expand_dims(x1, axis=axis)
    x2 = np.expand_dims(x2, axis=axis)
    return r * x1 + np.sqrt(1 - r**2) * x2