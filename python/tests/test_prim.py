import copy
import pytest
import libcasm.xtal as xtal


def test_make_primitive_occ(nonprimitive_cubic_occ_prim):
    assert nonprimitive_cubic_occ_prim.coordinate_frac().shape[1] == 2
    prim = xtal.make_primitive(nonprimitive_cubic_occ_prim)
    assert prim.coordinate_frac().shape[1] == 1


def test_make_primitive_manydof(test_nonprimitive_manydof_prim):
    assert test_nonprimitive_manydof_prim.coordinate_frac().shape[1] == 2
    prim = xtal.make_primitive(test_nonprimitive_manydof_prim)
    assert prim.coordinate_frac().shape[1] == 1


def test_asymmetric_unit_indices(perovskite_occ_prim):
    asymmetric_unit_indices = xtal.asymmetric_unit_indices(perovskite_occ_prim)
    assert len(asymmetric_unit_indices) == 3
    assert [0] in asymmetric_unit_indices
    assert [1] in asymmetric_unit_indices
    assert [2, 3, 4] in asymmetric_unit_indices


def test_simple_cubic_binary_factor_group(simple_cubic_binary_prim):
    prim = simple_cubic_binary_prim
    factor_group = xtal.make_factor_group(prim)
    assert len(factor_group) == 48


def test_simple_cubic_ising_factor_group(simple_cubic_ising_prim):
    prim = simple_cubic_ising_prim
    factor_group = xtal.make_factor_group(prim)
    assert len(factor_group) == 96


def test_simple_cubic_1d_disp_factor_group(simple_cubic_1d_disp_prim):
    prim = simple_cubic_1d_disp_prim
    factor_group = xtal.make_factor_group(prim)
    assert len(factor_group) == 16

def test_is_same_prim(simple_cubic_1d_disp_prim, simple_cubic_binary_prim):
    prim = simple_cubic_1d_disp_prim
    prim2 = simple_cubic_binary_prim

    assert prim is not prim2
    assert prim != prim2
    assert xtal._xtal._is_same_prim(prim, prim2) == False

    other = prim
    assert other is prim
    assert other == prim
    assert xtal._xtal._is_same_prim(other, prim)

    first = xtal._xtal._share_prim(prim)
    assert first is prim
    assert first == prim
    assert xtal._xtal._is_same_prim(first, prim)

    first = xtal._xtal._copy_prim(prim)
    assert first is not prim
    assert first != prim
    assert xtal._xtal._is_same_prim(first, prim) == False

    second = xtal._xtal._share_prim(prim2)
    assert second is not first
    assert second != first
    assert xtal._xtal._is_same_prim(second, first) == False
