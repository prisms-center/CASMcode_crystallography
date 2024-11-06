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


def test_construct_from_lattice_parameters():
    from math import degrees

    a = 1.0
    b = 1.3
    c = 1.6
    alpha = 90
    beta = 120
    gamma = 130
    lengths_and_angles = [a, b, c, alpha, beta, gamma]
    lat = xtal.Lattice.from_lengths_and_angles(lengths_and_angles)
    assert isinstance(lat, xtal.Lattice)

    L = lat.column_vector_matrix()

    assert math.isclose(np.linalg.norm(L[:, 0]), a)
    assert math.isclose(np.linalg.norm(L[:, 1]), b)
    assert math.isclose(np.linalg.norm(L[:, 2]), c)
    assert math.isclose(degrees(np.arccos(np.dot(L[:, 0], L[:, 1]) / (a * b))), gamma)
    assert math.isclose(degrees(np.arccos(np.dot(L[:, 0], L[:, 2]) / (a * c))), beta)
    assert math.isclose(degrees(np.arccos(np.dot(L[:, 1], L[:, 2]) / (b * c))), alpha)

    assert np.allclose(
        np.array(lengths_and_angles),
        np.array(lat.lengths_and_angles()),
    )


def test_repiprocal():
    x = 1.0 / math.sqrt(2.0)
    L = np.array(
        [
            [0.0, x, x],
            [x, 0.0, x],
            [x, x, 0.0],
        ]
    ).transpose()
    fcc_lattice = xtal.Lattice(L)

    vol = np.dot(L[:, 0], np.cross(L[:, 1], L[:, 2]))
    b1 = np.cross(L[:, 1], L[:, 2]) * (2 * math.pi) / vol
    b2 = np.cross(L[:, 2], L[:, 0]) * (2 * math.pi) / vol
    b3 = np.cross(L[:, 0], L[:, 1]) * (2 * math.pi) / vol

    assert np.isclose(fcc_lattice.volume(), vol)

    reciprocal_lattice = fcc_lattice.reciprocal()
    L_recip = reciprocal_lattice.column_vector_matrix()

    assert np.isclose(reciprocal_lattice.volume(), ((2 * math.pi) ** 3) / vol)
    assert np.allclose(L_recip[:, 0], b1)
    assert np.allclose(L_recip[:, 1], b2)
    assert np.allclose(L_recip[:, 2], b3)


def test_Lattice_to_from_dict():
    lattice = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],  # a
                [-0.3, 1.0, 0.0],  # b
                [0.2, 0.4, 1.0],  # c
            ]
        ).transpose()
    )

    data = lattice.to_dict()

    assert "lattice_vectors" in data
    assert np.allclose(
        np.array(data["lattice_vectors"]).transpose(), lattice.column_vector_matrix()
    )

    lattice2 = xtal.Lattice.from_dict(data)
    assert isinstance(lattice2, xtal.Lattice)
    assert lattice == lattice2


def test_Lattice_copy():
    import copy

    obj = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],  # a
                [-0.3, 1.0, 0.0],  # b
                [0.2, 0.4, 1.0],  # c
            ]
        ).transpose()
    )

    obj1 = obj.copy()
    assert isinstance(obj1, xtal.Lattice)
    assert obj1 is not obj

    obj2 = copy.copy(obj)
    assert isinstance(obj2, xtal.Lattice)
    assert obj2 is not obj

    obj3 = copy.deepcopy(obj)
    assert isinstance(obj3, xtal.Lattice)
    assert obj3 is not obj


def test_Lattice_repr():
    import io
    from contextlib import redirect_stdout

    lattice = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],  # a
                [-0.3, 1.0, 0.0],  # b
                [0.2, 0.4, 1.0],  # c
            ]
        ).transpose()
    )

    f = io.StringIO()
    with redirect_stdout(f):
        print(lattice)
    out = f.getvalue()
    assert "lattice_vectors" in out


def test_voronoi():
    ### Simple cubic lattice test ###
    x = 1.0
    L = np.array(
        [
            [x, 0.0, 0.0],
            [0.0, x, 0.0],
            [0.0, 0.0, x],
        ]
    ).transpose()
    cubic_lattice = xtal.Lattice(L)

    voronoi_table = cubic_lattice.voronoi_table()
    assert isinstance(voronoi_table, np.ndarray)
    assert voronoi_table.shape == (26, 3)
    assert cubic_lattice.voronoi_number([0.4, 0.2, 0.1]) == 0
    assert cubic_lattice.voronoi_number([0.5, 0.2, 0.1]) == 1
    assert cubic_lattice.voronoi_number([0.5, 0.5, 0.1]) == 3
    assert cubic_lattice.voronoi_number([0.5, 0.5, 0.5]) == 7
    assert cubic_lattice.voronoi_number([0.6, 0.2, 0.1]) == -1
    assert np.isclose(cubic_lattice.voronoi_inner_radius(), 0.5)

    ### FCC lattice test ###
    x = 1.0 / math.sqrt(2.0)
    L = np.array(
        [
            [0.0, x, x],
            [x, 0.0, x],
            [x, x, 0.0],
        ]
    ).transpose()
    fcc_lattice = xtal.Lattice(L)

    voronoi_table = fcc_lattice.voronoi_table()
    assert isinstance(voronoi_table, np.ndarray)
    assert voronoi_table.shape == (18, 3)
    assert fcc_lattice.voronoi_number([0.0, x / 2.0 - 0.1, x / 2.0 - 0.1]) == 0
    assert fcc_lattice.voronoi_number([0.0, x / 2.0, x / 2.0]) == 1
    assert fcc_lattice.voronoi_number([0.0, x / 2.0 + 0.1, x / 2.0 + 0.1]) == -1
    assert fcc_lattice.voronoi_number([x / 2.0, x / 2.0, x / 2.0]) == 3
    assert fcc_lattice.voronoi_number([0.0, x, 0.0]) == 5
    assert np.isclose(fcc_lattice.voronoi_inner_radius(), 0.5)

    ### BCC lattice test ###
    x = 1.0 / math.sqrt(3.0)
    L = np.array(
        [
            [-x, x, x],
            [x, -x, x],
            [x, x, -x],
        ]
    ).transpose()
    bcc_lattice = xtal.Lattice(L)

    voronoi_table = bcc_lattice.voronoi_table()
    assert isinstance(voronoi_table, np.ndarray)
    assert voronoi_table.shape == (14, 3)
    assert (
        bcc_lattice.voronoi_number([x / 2.0 - 0.1, x / 2.0 - 0.1, x / 2.0 - 0.1]) == 0
    )
    assert bcc_lattice.voronoi_number([x / 2.0, x / 2.0, x / 2.0]) == 1
    assert (
        bcc_lattice.voronoi_number([x / 2.0 + 0.1, x / 2.0 + 0.1, x / 2.0 + 0.1]) == -1
    )
    assert bcc_lattice.voronoi_number([x, 0.0, 0.0]) == 1
    assert bcc_lattice.voronoi_number([x / 2, x, 0.0]) == 3
    assert np.isclose(bcc_lattice.voronoi_inner_radius(), 0.5)
