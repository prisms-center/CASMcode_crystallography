import pytest

def test_asymmetric_unit_indices(perovskite_occ_prim):
    asymmetric_unit_indices = perovskite_occ_prim.asymmetric_unit_indices()
    assert len(asymmetric_unit_indices) == 3
    assert [0] in asymmetric_unit_indices
    assert [1] in asymmetric_unit_indices
    assert [2, 3, 4] in asymmetric_unit_indices
