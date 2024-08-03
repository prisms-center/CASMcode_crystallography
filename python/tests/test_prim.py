import json
import os

import numpy as np

import libcasm.xtal as xtal


def lial_lattice_and_coords() -> tuple[np.array, np.array]:
    """Populate lattice and coordinates
    of lial to compare

    Returns
    -------
    tuple[np.array, np.array]
        Tuple of lattice and fractional coordinates

    """
    lial_lattice = np.array(
        [
            [4.471006, 0.000000, -2.235503],
            [0.000000, 1.411149, -9.345034],
            [0.000000, 5.179652, 0.000000],
        ]
    )

    lial_frac_coords = np.array(
        [
            [
                0.232895,
                0.307941,
                0.15082,
                0.84918,
            ],
            [
                0.842594,
                0.472885,
                0.216684,
                0.783316,
            ],
            [
                0.46579,
                0.615882,
                0.30164,
                0.69836,
            ],
        ]
    )
    return lial_lattice, lial_frac_coords


def check_lial(prim):
    lattice, frac_coords = lial_lattice_and_coords()
    assert (
        np.allclose(lattice, prim.lattice().column_vector_matrix(), 1e-4, 1e-4) is True
    )
    assert np.allclose(frac_coords, prim.coordinate_frac(), 1e-4, 1e-4) is True
    assert prim.occ_dof() == [["Li"], ["Li"], ["Al"], ["Al"]]
    assert prim.labels() == [-1] * 4


def check_lial_with_occ_dofs(prim_with_occ_dofs, occ_dofs):
    lattice, frac_coords = lial_lattice_and_coords()
    assert (
        np.allclose(
            lattice, prim_with_occ_dofs.lattice().column_vector_matrix(), 1e-4, 1e-4
        )
        is True
    )
    assert (
        np.allclose(frac_coords, prim_with_occ_dofs.coordinate_frac(), 1e-4, 1e-4)
        is True
    )
    assert prim_with_occ_dofs.occ_dof() == occ_dofs
    assert prim_with_occ_dofs.labels() == [-1] * 4


def test_prim_from_poscar(shared_datadir):
    poscar_path = os.path.join(shared_datadir, "lial.vasp")

    # with no occ dofs
    prim = xtal.Prim.from_poscar(poscar_path)
    check_lial(prim)


def test_prim_from_poscar_with_occ_dof(shared_datadir):
    poscar_path = os.path.join(shared_datadir, "lial.vasp")

    # with occ dofs
    occ_dofs = [["Li", "Va"], ["Li"], ["Al", "Va"], ["Li", "Al"]]
    prim = xtal.Prim.from_poscar(poscar_path, occ_dofs)
    check_lial_with_occ_dofs(prim, occ_dofs)


def test_prim_from_poscar_str(shared_datadir):
    poscar_path = os.path.join(shared_datadir, "lial.vasp")

    with open(poscar_path, "r") as f:
        poscar_str = f.read()

    # with no occ dofs
    prim = xtal.Prim.from_poscar_str(poscar_str)
    check_lial(prim)


def test_prim_from_poscar_str_with_occ_dof(shared_datadir):
    poscar_path = os.path.join(shared_datadir, "lial.vasp")

    with open(poscar_path, "r") as f:
        poscar_str = f.read()

    # with occ dofs
    occ_dofs = [["Li", "Va"], ["Li"], ["Al", "Va"], ["Li", "Al"]]
    prim = xtal.Prim.from_poscar_str(poscar_str, occ_dofs)
    check_lial_with_occ_dofs(prim, occ_dofs)


def test_prim_from_poscar_str_selectivedynamics():
    poscar_str = """test prim
    1.0
    4.0 0.0 0.0
    0.0 4.0 0.0
    0.0 0.0 4.0
    A B
    1 3
    Selective dynamics
    Cartesian
    0.0 0.0 0.0 T T T
    0.5 0.5 0.5 T T T
    0.0 0.0 1.0 F F F
    0.5 0.5 1.5 T T T
    """
    prim = xtal.Prim.from_poscar_str(poscar_str)
    assert prim.occ_dof() == [["A"], ["B.1"], ["B.2"], ["B.1"]]

    occ_props = prim.occupants()["A"].properties()
    assert "selectivedynamics" in occ_props
    assert np.allclose(occ_props["selectivedynamics"], np.array([1.0, 1.0, 1.0]))

    occ_props = prim.occupants()["B.1"].properties()
    assert "selectivedynamics" in occ_props
    assert np.allclose(occ_props["selectivedynamics"], np.array([1.0, 1.0, 1.0]))

    occ_props = prim.occupants()["B.2"].properties()
    assert "selectivedynamics" in occ_props
    assert np.allclose(occ_props["selectivedynamics"], np.array([0.0, 0.0, 0.0]))

    assert prim.coordinate_frac().shape == (3, 4)
    assert prim.local_dof() == [[], [], [], []]


def test_prim_to_poscar_str(shared_datadir):
    poscar_path = os.path.join(shared_datadir, "lial.vasp")
    prim = xtal.Prim.from_poscar(poscar_path)
    structure = xtal.Structure(
        lattice=prim.lattice(),
        atom_coordinate_frac=prim.coordinate_frac(),
        atom_type=[x[0] for x in prim.occ_dof()],
    )
    lines = structure.to_poscar_str(title="Li2Al2").split("\n")
    assert lines[0] == "Li2Al2"
    assert lines[5] == "Al Li "
    assert lines[6] == "2 2 "
    assert lines[7] == "Direct"


def test_make_primitive_occ(nonprimitive_cubic_occ_prim):
    assert nonprimitive_cubic_occ_prim.coordinate_frac().shape[1] == 2
    prim = xtal.make_primitive(nonprimitive_cubic_occ_prim)
    assert isinstance(prim, xtal.Prim)
    assert prim.coordinate_frac().shape[1] == 1


def test_make_primitive_manydof(test_nonprimitive_manydof_prim):
    assert test_nonprimitive_manydof_prim.coordinate_frac().shape[1] == 2
    prim = xtal.make_primitive(test_nonprimitive_manydof_prim)
    assert isinstance(prim, xtal.Prim)
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
    assert xtal._xtal._is_same_prim(prim, prim2) is False

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
    assert xtal._xtal._is_same_prim(first, prim) is False

    second = xtal._xtal._share_prim(prim2)
    assert second is not first
    assert second != first
    assert xtal._xtal._is_same_prim(second, first) is False


def test_copy(simple_cubic_binary_prim):
    import copy

    prim = simple_cubic_binary_prim
    prim1 = prim.copy()
    assert isinstance(prim1, xtal.Prim)
    assert prim1 is not prim

    prim2 = copy.copy(prim)
    assert isinstance(prim2, xtal.Prim)
    assert prim2 is not prim

    prim3 = copy.deepcopy(prim)
    assert isinstance(prim3, xtal.Prim)
    assert prim3 is not prim


def test_to_dict(simple_cubic_binary_va_disp_Hstrain_prim):
    prim = simple_cubic_binary_va_disp_Hstrain_prim

    # convert to dict
    data = prim.to_dict()

    assert "lattice_vectors" in data
    assert "basis" in data
    assert len(data["basis"]) == 1
    assert "dofs" in data["basis"][0]
    assert "disp" in data["basis"][0]["dofs"]
    assert "coordinate_mode" in data
    assert "dofs" in data
    assert "Hstrain" in data["dofs"]


def test_from_dict():
    L1 = np.array(
        [
            [1.0, 0.0, 0.0],  # v1
            [-0.5, 1.0, 0.0],  # v2
            [0.0, 0.0, 2.0],  # v3
        ]
    ).transpose()
    basis_frac = np.array(
        [
            [0.0, 0.0, 0.0],  # b1
        ]
    ).transpose()
    data = {
        "title": "test",
        "lattice_vectors": L1.transpose().tolist(),
        "coordinate_mode": "Fractional",
        "basis": [
            {
                "coordinate": basis_frac[:, 0].tolist(),
                "occupants": ["A", "B", "Va"],
                "dofs": {"disp": {}},
            },
        ],
        "dofs": {"Hstrain": {}},
    }
    prim = xtal.Prim.from_dict(data)

    assert np.allclose(prim.lattice().column_vector_matrix(), L1)
    assert np.allclose(prim.coordinate_frac(), basis_frac)
    assert prim.occ_dof() == [["A", "B", "Va"]]

    prim_local_dof = prim.local_dof()
    assert len(prim_local_dof) == 1
    assert len(prim_local_dof[0]) == 1
    assert prim_local_dof[0][0].dofname() == "disp"

    prim_global_dof = prim.global_dof()
    assert len(prim_global_dof) == 1
    assert prim_global_dof[0].dofname() == "Hstrain"


def test_repr(simple_cubic_binary_va_disp_Hstrain_prim):
    import io
    from contextlib import redirect_stdout

    prim = simple_cubic_binary_va_disp_Hstrain_prim

    f = io.StringIO()
    with redirect_stdout(f):
        print(prim)
    out = f.getvalue()
    assert "basis" in out


def test_to_json(simple_cubic_binary_va_disp_Hstrain_prim):
    prim = simple_cubic_binary_va_disp_Hstrain_prim

    # convert to json string
    json_str = prim.to_json()

    data = json.loads(json_str)
    assert "lattice_vectors" in data
    assert "basis" in data
    assert len(data["basis"]) == 1
    assert "dofs" in data["basis"][0]
    assert "disp" in data["basis"][0]["dofs"]
    assert "coordinate_mode" in data
    assert "dofs" in data
    assert "Hstrain" in data["dofs"]


def test_from_json():
    L1 = np.array(
        [
            [1.0, 0.0, 0.0],  # v1
            [-0.5, 1.0, 0.0],  # v2
            [0.0, 0.0, 2.0],  # v3
        ]
    ).transpose()
    basis_frac = np.array(
        [
            [0.0, 0.0, 0.0],  # b1
        ]
    ).transpose()
    data = {
        "title": "test",
        "lattice_vectors": L1.transpose().tolist(),
        "coordinate_mode": "Fractional",
        "basis": [
            {
                "coordinate": basis_frac[:, 0].tolist(),
                "occupants": ["A", "B", "Va"],
                "dofs": {"disp": {}},
            },
        ],
        "dofs": {"Hstrain": {}},
    }

    json_str = json.dumps(data)

    prim = xtal.Prim.from_json(json_str)

    assert np.allclose(prim.lattice().column_vector_matrix(), L1)
    assert np.allclose(prim.coordinate_frac(), basis_frac)
    assert prim.occ_dof() == [["A", "B", "Va"]]

    prim_local_dof = prim.local_dof()
    assert len(prim_local_dof) == 1
    assert len(prim_local_dof[0]) == 1
    assert prim_local_dof[0][0].dofname() == "disp"

    prim_global_dof = prim.global_dof()
    assert len(prim_global_dof) == 1
    assert prim_global_dof[0].dofname() == "Hstrain"


def test_prim_with_labels():
    lattice = xtal.Lattice(np.eye(3))
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    ).transpose()
    occ_dof = [
        ["A", "B"],
        ["B", "A"],
        ["A", "B"],
        ["A", "B"],
    ]

    ## No basis site labels
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
    )
    factor_group = xtal.make_factor_group(prim)
    assert len(factor_group) == 48 * 4
    assert prim.labels() == [-1] * 4
    data = prim.to_dict()
    assert len(data["basis"]) == 4
    for site in data["basis"]:
        assert "label" not in site

    ## Add basis site labels
    prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        labels=[0, 1, 1, 1],
    )
    factor_group = xtal.make_factor_group(prim)
    assert len(factor_group) == 48
    assert prim.labels() == [0, 1, 1, 1]

    data = prim.to_dict()
    assert len(data["basis"]) == 4
    for site in data["basis"]:
        assert "label" in site
