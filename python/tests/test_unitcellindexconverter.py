import numpy as np
import pytest

import libcasm.xtal as xtal


def test_UnitCellIndexConverter_1():
    # test constructor
    f = xtal.UnitCellIndexConverter(
        np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        )
    )

    # test total_unitcells
    assert f.total_unitcells() == 8

    # test make_lattice_points
    assert len(f.make_lattice_points()) == 8

    # test (i,j,k) -> linear_unitcell_index
    assert f.linear_unitcell_index(np.array([0, 0, 0])) == 0
    assert f.linear_unitcell_index(np.array([1, 0, 0])) == 1
    assert f.linear_unitcell_index(np.array([0, 1, 0])) == 2
    assert f.linear_unitcell_index(np.array([1, 1, 0])) == 3
    assert f.linear_unitcell_index(np.array([0, 0, 1])) == 4
    assert f.linear_unitcell_index(np.array([1, 0, 1])) == 5
    assert f.linear_unitcell_index(np.array([0, 1, 1])) == 6
    assert f.linear_unitcell_index(np.array([1, 1, 1])) == 7

    # test linear_unitcell_index -> (i,j,k)
    assert (f.unitcell(0) == np.array([0, 0, 0])).all()
    assert (f.unitcell(1) == np.array([1, 0, 0])).all()
    assert (f.unitcell(2) == np.array([0, 1, 0])).all()
    assert (f.unitcell(3) == np.array([1, 1, 0])).all()
    assert (f.unitcell(4) == np.array([0, 0, 1])).all()
    assert (f.unitcell(5) == np.array([1, 0, 1])).all()
    assert (f.unitcell(6) == np.array([0, 1, 1])).all()
    assert (f.unitcell(7) == np.array([1, 1, 1])).all()

    # test explicit bring_within
    assert (f.bring_within(np.array([2, 2, 2])) == np.array([0, 0, 0])).all()

    # test default bring within
    assert f.linear_unitcell_index(np.array([2, 2, 2])) == 0

    # test never_bring_within
    with pytest.raises(RuntimeError):
        f.never_bring_within()
        assert f.linear_unitcell_index(np.array([2, 2, 2])) == 0

    # test always_bring_within
    f.never_bring_within()
    f.always_bring_within()
    assert f.linear_unitcell_index(np.array([2, 2, 2])) == 0


def test_UnitCellIndexConverter_2():
    # test constructor
    with pytest.raises(TypeError):
        # test constructor
        xtal.UnitCellIndexConverter(
            np.array(
                [
                    [2.0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                ]
            )
        )
