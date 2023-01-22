import copy
import json
import pytest
import libcasm.xtal as xtal
import numpy as np


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


def test_to_dict(simple_cubic_binary_va_disp_Hstrain_prim):
    prim = simple_cubic_binary_va_disp_Hstrain_prim

    # convert to dict
    data = prim.to_dict()

    assert 'lattice_vectors' in data
    assert 'basis' in data
    assert len(data['basis']) == 1
    assert 'dofs' in data['basis'][0]
    assert 'disp' in data['basis'][0]['dofs']
    assert 'coordinate_mode' in data
    assert 'dofs' in data
    assert 'Hstrain' in data['dofs']


def test_from_dict():
    L1 = np.array([
        [1.0, 0.0, 0.0],  # v1
        [-0.5, 1.0, 0.0],  # v2
        [0.0, 0.0, 2.0],  # v3
    ]).transpose()
    basis_frac = np.array([
        [0.0, 0.0, 0.0],  # b1
    ]).transpose()
    data = {
        'title':
        'test',
        'lattice_vectors':
        L1.transpose().tolist(),
        'coordinate_mode':
        'Fractional',
        'basis': [
            {
                'coordinate': basis_frac[:, 0].tolist(),
                'occupants': ['A', 'B', 'Va'],
                'dofs': {
                    'disp': {}
                },
            },
        ],
        'dofs': {
            'Hstrain': {}
        }
    }
    prim = xtal.Prim.from_dict(data)

    assert np.allclose(prim.lattice().column_vector_matrix(), L1)
    assert np.allclose(prim.coordinate_frac(), basis_frac)
    assert prim.occ_dof() == [['A', 'B', 'Va']]

    prim_local_dof = prim.local_dof()
    assert len(prim_local_dof) == 1
    assert len(prim_local_dof[0]) == 1
    assert prim_local_dof[0][0].dofname() == 'disp'

    prim_global_dof = prim.global_dof()
    assert len(prim_global_dof) == 1
    assert prim_global_dof[0].dofname() == 'Hstrain'


def test_to_json(simple_cubic_binary_va_disp_Hstrain_prim):
    prim = simple_cubic_binary_va_disp_Hstrain_prim

    # convert to json string
    json_str = prim.to_json()

    data = json.loads(json_str)
    assert 'lattice_vectors' in data
    assert 'basis' in data
    assert len(data['basis']) == 1
    assert 'dofs' in data['basis'][0]
    assert 'disp' in data['basis'][0]['dofs']
    assert 'coordinate_mode' in data
    assert 'dofs' in data
    assert 'Hstrain' in data['dofs']


def test_from_json():
    L1 = np.array([
        [1.0, 0.0, 0.0],  # v1
        [-0.5, 1.0, 0.0],  # v2
        [0.0, 0.0, 2.0],  # v3
    ]).transpose()
    basis_frac = np.array([
        [0.0, 0.0, 0.0],  # b1
    ]).transpose()
    data = {
        'title':
        'test',
        'lattice_vectors':
        L1.transpose().tolist(),
        'coordinate_mode':
        'Fractional',
        'basis': [
            {
                'coordinate': basis_frac[:, 0].tolist(),
                'occupants': ['A', 'B', 'Va'],
                'dofs': {
                    'disp': {}
                },
            },
        ],
        'dofs': {
            'Hstrain': {}
        }
    }

    json_str = json.dumps(data)

    prim = xtal.Prim.from_json(json_str)

    assert np.allclose(prim.lattice().column_vector_matrix(), L1)
    assert np.allclose(prim.coordinate_frac(), basis_frac)
    assert prim.occ_dof() == [['A', 'B', 'Va']]

    prim_local_dof = prim.local_dof()
    assert len(prim_local_dof) == 1
    assert len(prim_local_dof[0]) == 1
    assert prim_local_dof[0][0].dofname() == 'disp'

    prim_global_dof = prim.global_dof()
    assert len(prim_global_dof) == 1
    assert prim_global_dof[0].dofname() == 'Hstrain'
