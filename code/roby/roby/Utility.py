'''
The **Utilitys** module contains some utility function definitions

In particular, the utilities are those for
- Robustness approximation
- Exact parabola computation
- Parabola approximation

@author: Andrea Bombarda
'''

from sympy import solve   # type: ignore
from sympy.abc import a, b, c   # type: ignore
import numpy as np   # type: ignore


def compute_parabola(x1: float, y1: float, x2: float, y2: float, xv: float):
    """
    Computes the parameters a,b, and c of a parabola expressed in the shape of

    ax^2 + bx + c = y

    The computation is performed given the values of two points and the
    x-coordinate of the vertex

    Parameters
    ----------
        x1 : float
            x-coordinate of the first point
        y1 : float
            y-coordinate of the first point
        x2 : float
            x-coordinate of the second point
        y2 : float
            y-coordinate of the second point
        xv : float
            x-coordinate of the vertex of the parabola

    Returns
    -------
        parameters : List[float]
            list containing the parameters of the parabola in the
            shape of [a, b, c]
    """
    return solve([a*x1**2 + b*x1 + c - y1,
                  a*x2**2 + b*x2 + c - y2,
                  b**2-4*a*c + 4*a*xv], a, b, c)


def compute_appoximate_parabola(x1: float, y1: float, x2: float, y2: float,
                                x3: float, y3: float):
    """
    Computes the parameters a,b, and c of a parabola expressed in the shape of

    ax^2 + bx + c = y

    The computation is performed given the values of two points and the
    x-coordinate of the vertex

    Parameters
    ----------
        x1 : float
            x-coordinate of the first point
        y1 : float
            y-coordinate of the first point
        x2 : float
            x-coordinate of the second point
        y2 : float
            y-coordinate of the second point
        x3 : float
            x-coordinate of the third point
        y3 : float
            y-coordinate of the third point

    Returns
    -------
        parameters : List[float]
            list containing the parameters of the parabola in the
            shape of [a, b, c]
    """
    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    parameters = np.polyfit(x, y, 2)
    return parameters
