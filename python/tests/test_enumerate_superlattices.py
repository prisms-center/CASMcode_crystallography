import numpy as np

import libcasm.xtal as xtal


def test_enumerate_superlattices_simple_cubic_point_group():
    unit_lattice = xtal.Lattice(np.eye(3).transpose())
    point_group = xtal.make_point_group(unit_lattice)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc"
    )
    assert len(superlattices) == 16


def test_enumerate_superlattices_disp_1d_crystal_point_group(simple_cubic_1d_disp_prim):
    unit_lattice = xtal.Lattice(np.eye(3).transpose())
    point_group = xtal.make_crystal_point_group(simple_cubic_1d_disp_prim)
    superlattices = xtal.enumerate_superlattices(
        unit_lattice, point_group, max_volume=4, min_volume=1, dirs="abc"
    )
    assert len(superlattices) == 28
