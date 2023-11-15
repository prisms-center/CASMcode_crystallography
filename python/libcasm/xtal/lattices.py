import math
from typing import Optional

import numpy as np

import libcasm.xtal


def from_lattice_parameters(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> libcasm.xtal.Lattice:
    r"""Construct a lattice from lattice parameters

    Parameters
    ----------
    a: float
        Specify the lattice parameter :math:`a`.
    b: float
        Specify the lattice parameter :math:`b`.
    c: float
        Specify the lattice parameter :math:`c`.
    alpha: float
        Specify the lattice parameter :math:`\alpha`,
        in degrees.
    beta: float
        Specify the lattice parameter :math:`\beta`,
        in degrees.
    gamma: float
        Specify the lattice parameter :math:`\gamma`,
        in degrees.

    Returns
    -------
    lattice : xtal.Lattice
        A lattice with the specified lattice parameters
    """
    from math import cos, pow, radians, sin

    _alpha = radians(alpha)
    _beta = radians(beta)
    _gamma = radians(gamma)

    b_x = b * cos(_gamma)
    b_y = b * sin(_gamma)
    c_x = c * cos(_beta)

    # b_x * c_x + b_y * c_y = b * c * cos(_alpha)
    c_y = (b * c * cos(_alpha) - b_x * c_x) / b_y

    # c_y **2 + c_z **2 = pow(c * sin(_beta), 2.)
    c_z = math.sqrt(pow(c * sin(_beta), 2.0) - c_y * c_y)

    return libcasm.xtal.Lattice(
        np.array(
            [
                [a, 0.0, 0.0],
                [b_x, b_y, 0.0],
                [c_x, c_y, c_z],
            ]
        ).transpose()
    )


def cubic(a: float) -> libcasm.xtal.Lattice:
    r"""Construct a cubic lattice

    Parameters
    ----------
    a: float
        Specify the cubic lattice parameter :math:`a`.

    Returns
    -------
    lattice : xtal.Lattice
        A cubic lattice
    """
    return libcasm.xtal.Lattice(np.eye(3) * a)


def tetragonal(a: float, c: float) -> libcasm.xtal.Lattice:
    r"""Construct a tetragonal lattice

    Parameters
    ----------
    a: float
        Specify the tetragonal lattice parameter :math:`a`.
    c: float
        Specify the tetragonal lattice parameter :math:`c`.

    Returns
    -------
    lattice : xtal.Lattice
        A tetragonal lattice
    """
    return libcasm.xtal.Lattice(
        np.array(
            [
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0, 0.0, c],
            ]
        ).transpose()
    )


def hexagonal(a: float, c: float) -> libcasm.xtal.Lattice:
    r"""Construct a hexagonal lattice

    Parameters
    ----------
    a: float
        Specify the hexagonal lattice parameter :math:`a`.
    c: float
        Specify the hexagonal lattice parameter :math:`c`.

    Returns
    -------
    lattice : xtal.Lattice
        A hexagonal lattice
    """
    return libcasm.xtal.Lattice(
        np.array(
            [
                [a, 0.0, 0.0],  # a
                [-a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0],  # a
                [0, 0.0, c],
            ]
        ).transpose()
    )


def orthorhombic(a: float, b: float, c: float) -> libcasm.xtal.Lattice:
    r"""Construct an orthorhombic lattice

    Parameters
    ----------
    a: float
        Specify the orthorhombic lattice parameter :math:`a`.
    b: float
        Specify the orthorhombic lattice parameter :math:`b`.
    c: float
        Specify the orthorhombic lattice parameter :math:`c`.

    Returns
    -------
    lattice : xtal.Lattice
        A orthorhombic lattice
    """
    return libcasm.xtal.Lattice(
        np.array(
            [
                [a, 0.0, 0.0],
                [0.0, b, 0.0],
                [0, 0.0, c],
            ]
        ).transpose()
    )


def rhombohedral(a: float, alpha: float) -> libcasm.xtal.Lattice:
    r"""Construct a rhombohedral lattice

    Parameters
    ----------
    a: float
        Specify the rhombohedral lattice parameter :math:`a`.
    alpha: float
        Specify the rhombohedral lattice parameter :math:`\alpha`,
        in degrees.

    Returns
    -------
    lattice : xtal.Lattice
        A rhombohedral lattice
    """
    return from_lattice_parameters(a, a, a, alpha, alpha, alpha)


def monoclinic(a: float, b: float, c: float, beta: float) -> libcasm.xtal.Lattice:
    r"""Construct a monoclinic lattice

    Parameters
    ----------
    a: float
        Specify the monoclinic lattice parameter :math:`a`.
    b: float
        Specify the monoclinic lattice parameter :math:`b`.
    c: float
        Specify the monoclinic lattice parameter :math:`c`.
    beta: float
        Specify the monoclinic lattice parameter :math:`\beta`,
        in degrees.

    Returns
    -------
    lattice : xtal.Lattice
        A monoclinic lattice
    """
    return from_lattice_parameters(a, b, c, 90, beta, 90)


def triclinic(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> libcasm.xtal.Lattice:
    r"""Construct a triclinic lattice

    Parameters
    ----------
    a: float
        Specify the triclinic lattice parameter :math:`a`.
    b: float
        Specify the triclinic lattice parameter :math:`b`.
    c: float
        Specify the triclinic lattice parameter :math:`c`.
    alpha: float
        Specify the triclinic lattice parameter :math:`\alpha`,
        in degrees.
    beta: float
        Specify the triclinic lattice parameter :math:`\beta`,
        in degrees.
    gamma: float
        Specify the triclinic lattice parameter :math:`\gamma`,
        in degrees.

    Returns
    -------
    lattice : xtal.Lattice
        A triclinic lattice
    """
    return from_lattice_parameters(a, b, c, alpha, beta, gamma)


def BCC(r: Optional[float] = None, a: Optional[float] = None) -> libcasm.xtal.Lattice:
    r"""Construct a primitive BCC lattice

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct BCC primitive lattice
        with conventional lattice parameter :math:`a = 4r / \sqrt{3}`
    a: Optional[float] = None
        Specify the conventional BCC lattice parameter :math:`a`.

    Returns
    -------
    lattice : xtal.Lattice
        A primitive BCC lattice
    """
    if r is not None:
        a = 4.0 * r / math.sqrt(3.0)
    elif a is not None:
        r = a * math.sqrt(3.0) / 4.0
    else:
        raise Exception("Error in BCC: one of `r` or `a` is required")
    return libcasm.xtal.Lattice(
        np.array(
            [
                [-a / 2.0, a / 2.0, a / 2.0],  # a
                [a / 2.0, -a / 2.0, a / 2.0],  # a
                [a / 2.0, a / 2.0, -a / 2.0],  # a
            ]
        ).transpose()
    )


def FCC(r: Optional[float] = None, a: Optional[float] = None) -> libcasm.xtal.Lattice:
    r"""Construct a primitive FCC lattice

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct FCC primitive lattice
        with conventional lattice parameter :math:`a = 4r / \sqrt{2}`
    a: Optional[float] = None
        Specify the conventional FCC lattice parameter :math:`a`.

    Returns
    -------
    lattice : xtal.Lattice
        A primitive FCC lattice
    """
    if r is not None:
        a = 4.0 * r / math.sqrt(2.0)
    elif a is not None:
        r = a * math.sqrt(2.0) / 4.0
    else:
        raise Exception("Error in FCC: one of `r` or `a` is required")
    return libcasm.xtal.Lattice(
        np.array(
            [
                [0.0, a / 2.0, a / 2.0],
                [a / 2.0, 0.0, a / 2.0],
                [a / 2.0, a / 2.0, 0.0],
            ]
        ).transpose()
    )


def HCP(
    r: Optional[float] = None, a: Optional[float] = None, c: Optional[float] = None
) -> libcasm.xtal.Lattice:
    r"""Construct a primitive HCP lattice

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct HCP primitive lattice
        with conventional lattice parameter :math:`a = 2r`.
    a: Optional[float] = None
        Specify the conventional HCP lattice parameter :math:`a`.
    c: Optional[float] = None
        Specify the conventional HCP lattice parameter :math:`c`.
        If not specified, the ideal value is used :math:`c = a \sqrt{8/3}`.

    Returns
    -------
    lattice : xtal.Lattice
        A primitive HCP lattice
    """
    if r is not None:
        a = 2.0 * r
    elif a is not None:
        r = a * math.sqrt(2.0) / 4.0
    else:
        raise Exception("Error in HCP: one of `r` or `a` is required")

    if c is None:
        c = a * math.sqrt(8.0 / 3.0)

    return libcasm.xtal.Lattice(
        np.array(
            [
                [a, 0.0, 0.0],  # a
                [-a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0],  # a
                [0.0, 0.0, c],  # c
            ]
        ).transpose()
    )
