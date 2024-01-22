import numpy as np

import libcasm.xtal as xtal


def test_enumerate_superlattices_simple_cubic_point_group_1():
    unit_lattice = xtal.Lattice(np.eye(3).transpose())
    point_group = xtal.make_point_group(unit_lattice)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc"
    )
    assert len(superlattices) == 16


def test_enumerate_superlattices_simple_cubic_point_group_2():
    """Test diagonal_only parameter"""
    unit_lattice = xtal.Lattice(np.eye(3).transpose())
    point_group = xtal.make_point_group(unit_lattice)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice,
        point_group,
        max_volume=4,
        min_volume=1,
        dirs="abc",
        diagonal_only=True,
    )
    assert len(superlattices) == 5


def test_enumerate_superlattices_simple_cubic_point_group_3():
    """Test diagonal_only & fixed_shape parameters"""
    unit_lattice = xtal.Lattice(np.eye(3).transpose())
    point_group = xtal.make_point_group(unit_lattice)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice,
        point_group,
        max_volume=10,
        min_volume=1,
        dirs="abc",
        diagonal_only=True,
        fixed_shape=True,
    )
    assert len(superlattices) == 2


def test_enumerate_superlattices_simple_cubic_point_group_4():
    """Test unit_cell parameter"""
    unit_lattice = xtal.Lattice(np.eye(3).transpose())
    point_group = xtal.make_point_group(unit_lattice)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice,
        point_group,
        max_volume=4,
        min_volume=1,
        dirs="abc",
        unit_cell=np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    assert len(superlattices) == 26


def test_enumerate_superlattices_disp_1d_crystal_point_group(simple_cubic_1d_disp_prim):
    unit_lattice = xtal.Lattice(np.eye(3).transpose())
    point_group = xtal.make_crystal_point_group(simple_cubic_1d_disp_prim)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc"
    )
    assert len(superlattices) == 28
