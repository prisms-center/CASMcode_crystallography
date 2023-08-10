import math

import numpy as np

import libcasm.xtal.lattices as lattices
from libcasm.xtal import Lattice


def test_BCC_lattice():
    BCC_lattice = lattices.BCC(r=1.0)
    L = BCC_lattice.column_vector_matrix()
    a = 4.0 / math.sqrt(3.0)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert math.isclose(L[i, j], -a / 2.0)
            else:
                assert math.isclose(L[i, j], a / 2.0)


def test_from_lattice_parameters():
    from math import degrees

    a = 1.0
    b = 1.3
    c = 1.6
    alpha = 90
    beta = 120
    gamma = 130
    lat = lattices.from_lattice_parameters(
        a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
    )
    assert isinstance(lat, Lattice)

    L = lat.column_vector_matrix()

    assert math.isclose(np.linalg.norm(L[:, 0]), a)
    assert math.isclose(np.linalg.norm(L[:, 1]), b)
    assert math.isclose(np.linalg.norm(L[:, 2]), c)
    assert math.isclose(degrees(np.arccos(np.dot(L[:, 0], L[:, 1]) / (a * b))), gamma)
    assert math.isclose(degrees(np.arccos(np.dot(L[:, 0], L[:, 2]) / (a * c))), beta)
    assert math.isclose(degrees(np.arccos(np.dot(L[:, 1], L[:, 2]) / (b * c))), alpha)


def test_construct_all_lattices():
    assert isinstance(lattices.cubic(a=1.0), Lattice)
    assert isinstance(lattices.hexagonal(a=1.0, c=1.6), Lattice)
    assert isinstance(lattices.tetragonal(a=1.0, c=1.2), Lattice)
    assert isinstance(lattices.orthorhombic(a=1.0, b=1.3, c=1.6), Lattice)
    assert isinstance(lattices.monoclinic(a=1.0, b=1.3, c=1.6, beta=96.0), Lattice)
    assert isinstance(
        lattices.triclinic(a=1.0, b=1.3, c=1.6, alpha=78.0, beta=96.0, gamma=108.0),
        Lattice,
    )

    assert isinstance(lattices.BCC(r=1.0), Lattice)
    assert isinstance(lattices.BCC(a=1.0), Lattice)

    assert isinstance(lattices.FCC(r=1.0), Lattice)
    assert isinstance(lattices.FCC(a=1.0), Lattice)

    assert isinstance(lattices.HCP(r=1.0), Lattice)
    assert isinstance(lattices.HCP(r=1.0, c=2.0 * 1.8), Lattice)
    assert isinstance(lattices.HCP(a=1.0), Lattice)
    assert isinstance(lattices.HCP(a=1.0, c=1.8), Lattice)
