import os
import pytest
import numpy as np
import casm.xtal as xtal


@pytest.fixture
def pytest_root_dir(request: pytest.FixtureRequest) -> str:
    """Get pytest root dir (wherever pytest.ini/tox.ini/setup.cfg exists)
    Useful for resolving absolute paths of input files that are used in
    tests

    Parameters
    ----------
    request : pytest.FixtureRequest

    Returns
    -------
    str

    """
    return str(request.config.rootdir)


def test_prim_from_poscar(pytest_root_dir):
    li9al4_prim_path = os.path.join(
        pytest_root_dir, "tests", "input_files", "Li9Al4_primitive.vasp"
    )
    li9al4_prim = xtal.Prim.from_poscar(li9al4_prim_path)

    li9al4_lattice = np.array(
        [
            [4.471006, 0.000000, -2.235503],
            [0.000000, 1.411149, -9.345034],
            [0.000000, 5.179652, 0.000000],
        ]
    )
    li9al4_frac_coords = np.array(
        [
            [
                0.232895,
                0.307941,
                0.692059,
                0.543552,
                0.767105,
                0.456448,
                0.0,
                0.914098,
                0.085902,
                0.614294,
                0.385706,
                0.15082,
                0.84918,
            ],
            [
                0.842594,
                0.472885,
                0.527115,
                0.32586,
                0.157406,
                0.67414,
                0.0,
                0.359347,
                0.640653,
                0.934058,
                0.065942,
                0.216684,
                0.783316,
            ],
            [
                0.46579,
                0.615882,
                0.384118,
                0.087104,
                0.53421,
                0.912896,
                0.0,
                0.828196,
                0.171804,
                0.228588,
                0.771412,
                0.30164,
                0.69836,
            ],
        ]
    )
    assert (
        np.allclose(
            li9al4_lattice, li9al4_prim.lattice().column_vector_matrix(), 1e-4, 1e-4
        )
        is True
    )
    assert (
        np.allclose(li9al4_frac_coords, li9al4_prim.coordinate_frac(), 1e-4, 1e-4)
        is True
    )


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
    assert xtal._is_same_prim(prim, prim2) == False

    other = prim
    assert other is prim
    assert other == prim
    assert xtal._is_same_prim(other, prim)

    first = xtal._share_prim(prim)
    assert first is prim
    assert first == prim
    assert xtal._is_same_prim(first, prim)

    first = xtal._copy_prim(prim)
    assert first is not prim
    assert first != prim
    assert xtal._is_same_prim(first, prim) == False

    second = xtal._share_prim(prim2)
    assert second is not first
    assert second != first
    assert xtal._is_same_prim(second, first) == False
