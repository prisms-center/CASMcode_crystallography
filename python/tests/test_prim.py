import pytest


def test_make_primitive_occ(nonprimitive_cubic_occ_prim):
    assert nonprimitive_cubic_occ_prim.coordinate_frac().shape[1] == 2
    prim = nonprimitive_cubic_occ_prim.make_primitive()
    assert prim.coordinate_frac().shape[1] == 1


def test_make_primitive_manydof(test_nonprimitive_manydof_prim):
    assert test_nonprimitive_manydof_prim.coordinate_frac().shape[1] == 2
    prim = test_nonprimitive_manydof_prim.make_primitive()
    assert prim.coordinate_frac().shape[1] == 1


def test_asymmetric_unit_indices(perovskite_occ_prim):
    asymmetric_unit_indices = perovskite_occ_prim.asymmetric_unit_indices()
    assert len(asymmetric_unit_indices) == 3
    assert [0] in asymmetric_unit_indices
    assert [1] in asymmetric_unit_indices
    assert [2, 3, 4] in asymmetric_unit_indices


def test_simple_cubic_binary_factor_group(simple_cubic_binary_prim):
    prim = simple_cubic_binary_prim
    factor_group = prim.make_factor_group()
    assert len(factor_group) == 48


def test_simple_cubic_ising_factor_group(simple_cubic_ising_prim):
    prim = simple_cubic_ising_prim
    factor_group = prim.make_factor_group()
    assert len(factor_group) == 96


def test_simple_cubic_1d_disp_factor_group(simple_cubic_1d_disp_prim):
    prim = simple_cubic_1d_disp_prim
    factor_group = prim.make_factor_group()
    assert len(factor_group) == 16
