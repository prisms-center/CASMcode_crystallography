import math

import numpy as np

import libcasm.xtal as xtal


def test_tol():
    lattice = xtal.Lattice(np.eye(3))
    assert math.isclose(lattice.tol(), 1e-5)

    lattice = xtal.Lattice(np.eye(3), tol=1e-6)
    assert math.isclose(lattice.tol(), 1e-6)

    lattice.set_tol(1e-5)
    assert math.isclose(lattice.tol(), 1e-5)


def test_conversions(tetragonal_lattice):
    lattice = tetragonal_lattice
    assert lattice.column_vector_matrix().shape == (3, 3)

    coordinate_frac = np.array(
        [
            [0.0, 0.5, 0.5],
        ]
    ).transpose()
    coordinate_cart = np.array(
        [
            [0.0, 0.5, 1.0],
        ]
    ).transpose()

    assert np.allclose(
        xtal.fractional_to_cartesian(lattice, coordinate_frac), coordinate_cart
    )
    assert np.allclose(
        xtal.cartesian_to_fractional(lattice, coordinate_cart), coordinate_frac
    )

    coordinate_frac_outside = np.array(
        [
            [1.1, -0.1, 0.5],
        ]
    ).transpose()
    coordinate_frac_within = np.array(
        [
            [0.1, 0.9, 0.5],
        ]
    ).transpose()
    assert np.allclose(
        xtal.fractional_within(lattice, coordinate_frac_outside), coordinate_frac_within
    )


def test_min_periodic_displacement():
    lattice = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],  # a (along x)
                [0.0, 1.0, 0.0],  # a (along y)
                [0.0, 0.0, 1.0],  # a (along z)
            ]
        ).transpose()
    )
    r1 = np.array([0.1, 0.2, 0.9])
    r2 = np.array([0.1, 0.2, 0.1])
    d = xtal.min_periodic_displacement(lattice, r1, r2)
    assert np.allclose(d, np.array([0.0, 0.0, 0.2]))
    d_fast = xtal.min_periodic_displacement(lattice, r1, r2, robust=False)
    assert np.allclose(d_fast, np.array([0.0, 0.0, 0.2]))


def test_make_canonical():
    tetragonal_lattice_noncanonical = xtal.Lattice(
        np.array(
            [
                [0.0, 0.0, 2.0],  # c (along z)
                [1.0, 0.0, 0.0],  # a (along x)
                [0.0, 1.0, 0.0],  # a (along y)
            ]
        ).transpose()
    )
    lattice = xtal.make_canonical(tetragonal_lattice_noncanonical)
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array(
            [
                [1.0, 0.0, 0.0],  # a
                [0.0, 1.0, 0.0],  # a
                [0.0, 0.0, 2.0],  # c
            ]
        ).transpose(),
    )


def test_lattice_comparison():
    L1 = xtal.Lattice(
        np.array(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ).transpose()
    )
    L2 = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).transpose()
    )
    assert L1 < L2
    assert L1 <= L2
    assert L2 > L1
    assert L2 >= L1
    assert (L1 == L2) is False
    assert L1 != L2
    assert L1 == L1
    assert (L1 != L1) is False
    assert L1.is_equivalent_to(L2) is True


def test_is_superlattice_of():
    unit_lattice = xtal.Lattice(np.eye(3))
    lattice1 = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).transpose()
    )

    is_superlattice_of, T = lattice1.is_superlattice_of(unit_lattice)
    assert is_superlattice_of is True
    assert np.allclose(T, lattice1.column_vector_matrix())

    lattice2 = xtal.Lattice(lattice1.column_vector_matrix() * 2)
    is_superlattice_of, T = lattice2.is_superlattice_of(lattice1)
    assert is_superlattice_of is True
    assert np.allclose(T, np.eye(3) * 2)

    lattice3 = xtal.Lattice(
        np.array(
            [
                [4.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).transpose()
    )
    is_superlattice_of, T = lattice3.is_superlattice_of(lattice1)
    assert is_superlattice_of is False


def test_is_equivalent_superlattice_of():
    L = np.eye(3)

    S1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    ).transpose()

    S2 = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).transpose()

    unit_lattice = xtal.Lattice(L)
    point_group = xtal.make_point_group(unit_lattice)
    lattice1 = xtal.Lattice(S1)
    lattice2 = xtal.Lattice(S2)

    is_equivalent_superlattice_of, T, p = lattice2.is_equivalent_superlattice_of(
        lattice1, point_group
    )
    assert is_equivalent_superlattice_of is True
    assert np.allclose(S2, point_group[p].matrix() @ S1 @ T)
