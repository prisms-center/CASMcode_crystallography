import math
import pytest
import casm.xtal as xtal
import numpy as np

def test_tol():
    lattice = xtal.Lattice(np.eye(3).transpose())
    assert math.isclose(lattice.tol(), 1e-5)

    lattice = xtal.Lattice(np.eye(3).transpose(), tol=1e-6)
    assert math.isclose(lattice.tol(), 1e-6)

    lattice.set_tol(1e-5)
    assert math.isclose(lattice.tol(), 1e-5)

def test_conversions(tetragonal_lattice):
    lattice = tetragonal_lattice
    assert lattice.column_vector_matrix().shape == (3,3)

    coordinate_frac = np.array([[0.0, 0.5, 0.5]]).transpose()
    coordinate_cart = np.array([[0.0, 0.5, 1.0]]).transpose()

    assert np.allclose(lattice.fractional_to_cartesian(coordinate_frac), coordinate_cart)
    assert np.allclose(lattice.cartesian_to_fractional(coordinate_cart), coordinate_frac)

    coordinate_frac_outside = np.array([[1.1, -0.1, 0.5]]).transpose()
    coordinate_frac_within = np.array([[0.1, 0.9, 0.5]]).transpose()
    assert np.allclose(
        lattice.fractional_within(coordinate_frac_outside),
        coordinate_frac_within)

def test_make_canonical():
    tetragonal_lattice_noncanonical = xtal.Lattice(
        np.array([
            [0., 0., 2.], # c (along z)
            [1., 0., 0.], # a (along x)
            [0., 1., 0.]] # a (along y)
        ).transpose())
    lattice = tetragonal_lattice_noncanonical.make_canonical()
    assert np.allclose(
        lattice.column_vector_matrix(),
        np.array([
            [1., 0., 0.], # a
            [0., 1., 0.], # a
            [0., 0., 2.]] # c
        ).transpose())
