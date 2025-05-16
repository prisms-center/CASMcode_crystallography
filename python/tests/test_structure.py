import math

import numpy as np
import pytest

import libcasm.xtal as xtal
import libcasm.xtal.structures as xtal_structures


def test_make_structure(example_structure_1):
    structure = example_structure_1

    assert structure.lattice().column_vector_matrix().shape == (3, 3)
    assert structure.atom_coordinate_frac().shape == (3, 4)
    assert structure.atom_coordinate_cart().shape == (3, 4)
    assert len(structure.atom_type()) == 4

    assert len(structure.atom_properties()) == 1
    assert "disp" in structure.atom_properties()
    assert structure.atom_properties()["disp"].shape == (3, 4)

    assert len(structure.mol_properties()) == 0

    assert len(structure.global_properties()) == 1
    assert "Hstrain" in structure.global_properties()
    assert structure.global_properties()["Hstrain"].shape == (6, 1)


def test_make_structure_within():
    # Lattice vectors
    lattice = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],  # a
                [0.0, 1.0, 0.0],  # a
                [0.0, 0.0, 1.0],  # c
            ]
        ).transpose()
    )
    atom_coordinate_cart = np.array(
        [
            [0.0, 0.0, 1.1],
        ]
    ).transpose()

    init_structure = xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=xtal.fractional_to_cartesian(
            lattice, atom_coordinate_cart
        ),
        atom_type=["A"],
    )

    structure = xtal.make_structure_within(init_structure)
    expected_atom_coordinate_cart = np.array(
        [
            [0.0, 0.0, 0.1],
        ]
    ).transpose()
    assert np.allclose(structure.atom_coordinate_cart(), expected_atom_coordinate_cart)

    structure = xtal.make_within(init_structure)
    assert np.allclose(structure.atom_coordinate_cart(), expected_atom_coordinate_cart)


def test_structure_to_dict(example_structure_1):
    structure = example_structure_1
    data = structure.to_dict()

    assert "lattice_vectors" in data
    assert "coordinate_mode" in data
    assert len(data["atom_type"]) == 4
    assert "atom_coords" in data
    assert "atom_properties" in data
    assert len(data["atom_properties"]) == 1
    assert "disp" in data["atom_properties"]
    assert len(data["global_properties"]) == 1
    assert "Hstrain" in data["global_properties"]
    expected = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.25], [0.0, 0.0, 0.5], [0.5, 0.5, 0.75]]
    )
    print(xtal.pretty_json(data["atom_coords"]))
    assert np.allclose(np.array(data["atom_coords"]), expected)

    assert isinstance(xtal.pretty_json(data), str)

    data = structure.to_dict(excluded_species=["B"])
    assert data["atom_type"] == ["A", "A"]

    data = structure.to_dict(frac=False)
    expected = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [0.5, 0.5, 1.5]]
    )
    print(xtal.pretty_json(data["atom_coords"]))
    assert np.allclose(np.array(data["atom_coords"]), expected)


def test_structure_from_dict():
    data = {
        "atom_coords": [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 1.5],
        ],
        "atom_properties": {
            "disp": {
                "value": [
                    [0.1, 0.0, 0.0],
                    [0.1, 0.0, 0.1],
                    [0.1, 0.1, 0.0],
                    [0.1, 0.2, 0.3],
                ]
            }
        },
        "atom_type": ["A", "A", "B", "B"],
        "coordinate_mode": "Cartesian",
        "global_properties": {
            "Hstrain": {"value": [0.009950330853168087, 0.0, 0.0, 0.0, 0.0, 0.0]}
        },
        "lattice_vectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        "mol_coords": [],
        "mol_type": [],
    }
    structure = xtal.Structure.from_dict(data)

    assert structure.lattice().column_vector_matrix().shape == (3, 3)
    assert structure.atom_coordinate_frac().shape == (3, 4)
    assert structure.atom_coordinate_cart().shape == (3, 4)
    assert structure.mol_coordinate_cart().shape == (3, 0)
    assert structure.mol_coordinate_frac().shape == (3, 0)
    assert len(structure.atom_type()) == 4
    assert len(structure.atom_properties()) == 1
    assert len(structure.global_properties()) == 1


def test_structure_to_poscar_str_1(example_structure_2):
    poscar_str = example_structure_2.to_poscar_str(
        title="test structure", sort=False, cart_coordinate_mode=False
    )
    # print(poscar_str)
    lines = poscar_str.split("\n")

    assert lines[0] == "test structure"
    assert lines[5] == "A B A "
    assert lines[6] == "1 1 1 "
    assert lines[7] == "Direct"


def test_structure_to_poscar_str_2(example_structure_2):
    poscar_str = example_structure_2.to_poscar_str(
        title="test structure", sort=True, cart_coordinate_mode=True
    )
    # print(poscar_str)
    lines = poscar_str.split("\n")

    assert lines[0] == "test structure"
    assert lines[5] == "A B "
    assert lines[6] == "2 1 "
    assert lines[7] == "Cartesian"


def test_structure_to_poscar_str_3(example_structure_2):
    poscar_str = example_structure_2.to_poscar_str(
        title="test structure", sort=True, ignore=[]
    )
    print(poscar_str)
    lines = poscar_str.split("\n")

    assert lines[0] == "test structure"
    assert lines[5] == "A B Va "
    assert lines[6] == "2 1 1 "
    assert lines[7] == "Direct"


def test_structure_from_poscar_str_1():
    poscar_str = """test structure
    1.0
    4.0 0.0 0.0
    0.0 4.0 0.0
    0.0 0.0 4.0
    A B 
    1 3 
    Cartesian
    0.0 0.0 0.0 
    0.5 0.5 0.5 
    0.0 0.0 1.0 
    0.5 0.5 1.5 
    """
    structure = xtal.Structure.from_poscar_str(poscar_str)

    assert np.allclose(structure.lattice().column_vector_matrix(), np.eye(3) * 4.0)
    assert len(structure.mol_type()) == 0
    assert structure.mol_coordinate_frac().shape == (3, 0)
    assert len(structure.mol_properties()) == 0
    assert len(structure.atom_type()) == 4
    assert structure.atom_coordinate_frac().shape == (3, 4)
    assert len(structure.atom_properties()) == 0


def test_structure_from_poscar_str_with_selectivedynamics_1():
    poscar_str = """test structure
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
    # default, read as atoms
    structure = xtal.Structure.from_poscar_str(poscar_str)

    assert np.allclose(structure.lattice().column_vector_matrix(), np.eye(3) * 4.0)

    assert len(structure.mol_type()) == 0
    assert structure.mol_coordinate_frac().shape == (3, 0)
    assert len(structure.mol_properties()) == 0

    assert len(structure.atom_type()) == 4
    assert structure.atom_coordinate_frac().shape == (3, 4)
    assert len(structure.atom_properties()) == 1
    assert "selectivedynamics" in structure.atom_properties()
    assert structure.atom_properties()["selectivedynamics"].shape == (3, 4)


def test_structure_from_poscar_str_with_selectivedynamics_2():
    poscar_str = """test structure
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
    # default, read as molecules
    structure = xtal.Structure.from_poscar_str(poscar_str, mode="molecules")

    assert np.allclose(structure.lattice().column_vector_matrix(), np.eye(3) * 4.0)

    assert len(structure.mol_type()) == 4
    assert structure.mol_coordinate_frac().shape == (3, 4)
    assert len(structure.mol_properties()) == 1
    assert "selectivedynamics" in structure.mol_properties()
    assert structure.mol_properties()["selectivedynamics"].shape == (3, 4)

    assert len(structure.atom_type()) == 0
    assert structure.atom_coordinate_frac().shape == (3, 0)
    assert len(structure.atom_properties()) == 0


def test_structure_from_poscar_str_with_selectivedynamics_3():
    poscar_str = """test structure
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
    # default, read as atoms & molecules
    structure = xtal.Structure.from_poscar_str(poscar_str, mode="both")

    assert np.allclose(structure.lattice().column_vector_matrix(), np.eye(3) * 4.0)

    assert len(structure.mol_type()) == 4
    assert structure.mol_coordinate_frac().shape == (3, 4)
    assert len(structure.mol_properties()) == 1
    assert "selectivedynamics" in structure.mol_properties()
    assert structure.mol_properties()["selectivedynamics"].shape == (3, 4)

    assert len(structure.mol_type()) == 4
    assert structure.mol_coordinate_frac().shape == (3, 4)
    assert len(structure.mol_properties()) == 1
    assert "selectivedynamics" in structure.mol_properties()
    assert structure.mol_properties()["selectivedynamics"].shape == (3, 4)


def test_copy_structure(example_structure_1):
    import copy

    structure1 = copy.copy(example_structure_1)
    structure2 = copy.deepcopy(example_structure_1)
    structure3 = example_structure_1.copy()

    assert isinstance(example_structure_1, xtal.Structure)
    assert isinstance(structure1, xtal.Structure)
    assert structure1 is not example_structure_1
    assert isinstance(structure2, xtal.Structure)
    assert structure2 is not example_structure_1
    assert isinstance(structure3, xtal.Structure)
    assert structure3 is not example_structure_1


def test_Structure_repr(example_structure_1):
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        print(example_structure_1)
    out = f.getvalue()
    assert "atom_type" in out


def test_structure_is_equivalent_to():
    # Lattice vectors
    lattice = xtal.Lattice(
        np.array(
            [
                [1.0, 0.0, 0.0],  # a
                [0.0, 1.0, 0.0],  # a
                [0.0, 0.0, 2.0],  # c
            ]
        ).transpose()
    )
    atom_coordinate_cart = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 1.5],
        ]
    ).transpose()

    # atom types - re-ordered
    atom_type = ["A", "A", "B", "B"]

    # atom properties
    atom_disp = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.2, 0.3],
        ]
    ).transpose()
    atom_properties = {"disp": atom_disp}
    print(atom_properties)

    # global properties
    # F = np.array(
    #     [
    #         [1.01, 0.0, 0.0],
    #         [0.0, 1.0, 0.0],
    #         [0.0, 0.0, 1.0],
    #     ]
    # )
    # converter = xtal.StrainConverter('Hstrain')
    # Hstrain_vector = converter.from_F(F)
    Hstrain_vector = np.array([0.009950330853168087, 0.0, 0.0, 0.0, 0.0, 0.0])
    global_properties = {"Hstrain": Hstrain_vector}
    print(global_properties)

    structure1 = xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=xtal.cartesian_to_fractional(
            lattice, atom_coordinate_cart
        ),
        atom_type=atom_type,
        atom_properties=atom_properties,
        global_properties=global_properties,
    )

    # structure2: re-order atoms

    # atom coordinates - re-ordered
    atom_coordinate_cart = np.array(
        [
            [0.5, 0.5, 0.5],  # 1
            [0.0, 0.0, 1.0],  # 2
            [0.5, 0.5, 1.5],  # 3
            [0.0, 0.0, 0.0],  # 0
        ]
    ).transpose()

    # atom types - re-ordered
    atom_type = ["A", "B", "B", "A"]  # 1, 2, 3, 0

    # atom properties - re-ordered
    atom_disp = np.array(
        [
            [0.0, 0.1, 0.0],  # 1
            [0.0, 0.0, 0.1],  # 2
            [0.1, 0.2, 0.3],  # 3
            [0.1, 0.0, 0.0],  # 0
        ]
    ).transpose()
    atom_properties = {"disp": atom_disp}
    print(atom_properties)

    # global properties - same
    np.array(
        [
            [1.01, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # converter = xtal.StrainConverter('Hstrain')
    # Hstrain_vector = converter.from_F(F)
    Hstrain_vector = np.array([0.009950330853168087, 0.0, 0.0, 0.0, 0.0, 0.0])
    global_properties = {"Hstrain": Hstrain_vector}

    structure2 = xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=xtal.cartesian_to_fractional(
            lattice, atom_coordinate_cart
        ),
        atom_type=atom_type,
        atom_properties=atom_properties,
        global_properties=global_properties,
    )

    assert structure1.is_equivalent_to(structure2)


def test_make_superstructure_1():
    struc = xtal_structures.BCC(r=1)
    transformation_matrix = np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    assert transformation_matrix.dtype is np.dtype(np.float64)
    with pytest.raises(TypeError):
        xtal.make_superstructure(transformation_matrix, struc)


def test_make_superstructure_2():
    struc = xtal_structures.BCC(r=1)
    transformation_matrix = np.array([[0, 1, 1], [2, 0, 1], [1, 1, 0]], dtype=int).T
    assert transformation_matrix.flags.f_contiguous
    assert transformation_matrix.dtype is np.dtype(np.int64)
    superstruc = xtal.make_superstructure(transformation_matrix, struc)
    assert np.allclose(
        superstruc.lattice().column_vector_matrix(),
        struc.lattice().column_vector_matrix() @ transformation_matrix,
    )

    transformation_matrix_b = np.array([[0, 2, 1], [1, 0, 1], [1, 1, 0]], dtype=int)
    assert transformation_matrix_b.flags.c_contiguous
    assert transformation_matrix.dtype is np.dtype(np.int64)
    superstruc_b = xtal.make_superstructure(transformation_matrix_b, struc)
    assert np.allclose(
        superstruc_b.lattice().column_vector_matrix(),
        struc.lattice().column_vector_matrix() @ transformation_matrix_b,
    )

    assert np.allclose(
        transformation_matrix,
        transformation_matrix_b,
    )
    assert np.allclose(
        superstruc.lattice().column_vector_matrix(),
        superstruc_b.lattice().column_vector_matrix(),
    )


def test_make_superstructure_3(example_structure_1):
    struc = example_structure_1

    transformation_matrix = np.eye(3, dtype=int)

    superstructure = xtal.make_superstructure(
        transformation_matrix_to_super=transformation_matrix,
        structure=struc,
    )
    assert isinstance(superstructure, xtal.Structure)
    assert np.allclose(
        struc.atom_coordinate_cart(), superstructure.atom_coordinate_cart()
    )


def test_make_superstructure_4(example_structure_1):
    struc = example_structure_1

    transformation_matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 2],
        ],
        dtype=int,
    )

    superstructure = xtal.make_superstructure(
        transformation_matrix_to_super=transformation_matrix,
        structure=struc,
    )
    assert isinstance(superstructure, xtal.Structure)

    expected_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 2.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 3.5],
        ]
    ).transpose()
    assert np.allclose(superstructure.atom_coordinate_cart(), expected_coords)

    expected_disp = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ]
    ).transpose()
    assert np.allclose(superstructure.atom_properties()["disp"], expected_disp)

    assert np.allclose(
        struc.global_properties()["Hstrain"],
        superstructure.global_properties()["Hstrain"],
    )


def test_make_primitive_structure_1():
    struc = xtal_structures.BCC(r=1)
    transformation_matrix = np.array([[0, 1, 1], [2, 0, 1], [1, 1, 0]], dtype=int).T
    assert transformation_matrix.flags.f_contiguous
    assert transformation_matrix.dtype is np.dtype(np.int64)
    superstruc = xtal.make_superstructure(transformation_matrix, struc)
    assert np.allclose(
        superstruc.lattice().column_vector_matrix(),
        struc.lattice().column_vector_matrix() @ transformation_matrix,
    )

    primitive_struc = xtal.make_primitive_structure(superstruc)
    assert primitive_struc.is_equivalent_to(struc)


def test_make_primitive_structure_2():
    struc = xtal_structures.BCC(r=1)
    conventional_struc = xtal_structures.BCC(r=1, conventional=True)
    primitive_struc = xtal.make_primitive_structure(conventional_struc)
    assert primitive_struc.is_equivalent_to(struc)


def test_make_primitive_structure_3():
    struc = xtal_structures.FCC(r=1)
    conventional_struc = xtal_structures.FCC(r=1, conventional=True)
    primitive_struc = xtal.make_primitive_structure(conventional_struc)
    assert primitive_struc.is_equivalent_to(struc)


def test_make_canonical_structure_1():
    struc = xtal.Structure(
        lattice=xtal.Lattice(
            np.array(
                [
                    [0.0, 0.0, 1.0],  # z
                    [1.0, 0.0, 0.0],  # x
                    [0.0, 1.0, 0.0],  # y
                ]
            ).transpose()
        ),
        atom_coordinate_frac=np.array(
            [
                [0.0, 0.0, 0.0],
            ]
        ).transpose(),
        atom_type=["A"],
    )
    canonical_struc = xtal.make_canonical_structure(struc)
    assert np.allclose(canonical_struc.lattice().column_vector_matrix(), np.eye(3))


def test_make_superstructure_and_rotate():
    struc = xtal_structures.BCC(r=1)
    assert len(struc.atom_type()) == 1

    rotation_matrix = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 1],
        ]
    ).T
    transformation_matrix = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=int,
    ).T
    assert math.isclose(np.linalg.det(transformation_matrix), 2.0)

    superstruc = xtal.make_superstructure(transformation_matrix, struc)
    L = struc.lattice().column_vector_matrix()
    S = superstruc.lattice().column_vector_matrix()
    assert np.allclose(S, L @ transformation_matrix)
    assert len(superstruc.atom_type()) == 2

    symop = xtal.SymOp(
        matrix=rotation_matrix,
        translation=np.zeros((3,)),
        time_reversal=False,
    )
    rotated_superstruc = symop * superstruc

    L = struc.lattice().column_vector_matrix()
    S = rotated_superstruc.lattice().column_vector_matrix()
    assert np.allclose(S, rotation_matrix @ L @ transformation_matrix)
    assert len(rotated_superstruc.atom_type()) == 2


def test_structure_sort_structure_by_atom_type():
    struc = xtal_structures.BCC(a=1.0, conventional=True)
    superstruc = xtal.make_superstructure(np.eye(3, dtype=int) * 3, struc)
    unsorted_atom_type = ["A", "B"] * 27
    unsorted_struc = xtal.Structure(
        lattice=superstruc.lattice(),
        atom_type=unsorted_atom_type,
        atom_coordinate_frac=superstruc.atom_coordinate_frac(),
    )
    assert unsorted_struc.atom_type() == unsorted_atom_type
    # print(xtal.pretty_json(unsorted_struc.to_dict()))

    sorted_struc = xtal.sort_structure_by_atom_type(unsorted_struc)
    # print(xtal.pretty_json(sorted_struc.to_dict()))
    assert sorted_struc.atom_type() == ["A"] * 27 + ["B"] * 27


def test_structure_sort_structure_by_atom_coordinate_frac():
    struc = xtal_structures.BCC(r=1.0)
    unsorted_struc = xtal.make_superstructure(np.eye(3, dtype=int) * 2, struc)

    sorted_struc = xtal.sort_structure_by_atom_coordinate_frac(
        unsorted_struc,
        order="cba",
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
    ).transpose()
    # print(xtal.pretty_json(unsorted_struc.to_dict(frac=True)))
    assert np.allclose(sorted_struc.atom_coordinate_frac(), expected)

    sorted_struc = xtal.sort_structure_by_atom_coordinate_frac(
        unsorted_struc,
        order="abc",
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ]
    ).transpose()
    # print(xtal.pretty_json(sorted_struc.to_dict(frac=True)))
    assert np.allclose(sorted_struc.atom_coordinate_frac(), expected)


def test_structure_sort_structure_by_atom_coordinate_cart():
    struc = xtal_structures.BCC(a=1.0, conventional=True)
    unsorted_struc = xtal.make_superstructure(np.eye(3, dtype=int) * 2, struc)
    # print(xtal.pretty_json(unsorted_struc.to_dict()))

    sorted_struc = xtal.sort_structure_by_atom_coordinate_cart(
        unsorted_struc,
        order="zyx",
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
            [1.5, 1.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 1.5],
            [1.5, 0.5, 1.5],
            [0.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
        ]
    ).transpose()
    # print(xtal.pretty_json(sorted_struc.to_dict()))
    assert np.allclose(sorted_struc.atom_coordinate_cart(), expected)

    sorted_struc = xtal.sort_structure_by_atom_coordinate_cart(
        unsorted_struc,
        order="zyx",
        reverse=True,
    )
    expected = np.array(
        [
            [1.5, 1.5, 1.5],
            [0.5, 1.5, 1.5],
            [1.5, 0.5, 1.5],
            [0.5, 0.5, 1.5],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.5, 1.5, 0.5],
            [0.5, 1.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ).transpose()
    # print(xtal.pretty_json(sorted_struc.to_dict()))
    assert np.allclose(sorted_struc.atom_coordinate_cart(), expected)

    sorted_struc = xtal.sort_structure_by_atom_coordinate_cart(
        unsorted_struc,
        order="xyz",
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 1.5],
            [0.5, 1.5, 0.5],
            [0.5, 1.5, 1.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.5, 0.5, 0.5],
            [1.5, 0.5, 1.5],
            [1.5, 1.5, 0.5],
            [1.5, 1.5, 1.5],
        ]
    ).transpose()
    # print(xtal.pretty_json(sorted_struc.to_dict()))
    assert np.allclose(sorted_struc.atom_coordinate_cart(), expected)


def test_substitute_structure_species_1(example_structure_1):
    assert example_structure_1.atom_type() == ["A", "A", "B", "B"]
    s2 = xtal.substitute_structure_species(
        example_structure_1,
        {"A": "C"},
    )
    assert s2.atom_type() == ["C", "C", "B", "B"]


def test_substitute_structure_species_2(example_structure_1):
    assert example_structure_1.atom_type() == ["A", "A", "B", "B"]
    s2 = xtal.substitute_structure_species(
        example_structure_1,
        {"A": "C", "B": "D"},
    )
    assert s2.atom_type() == ["C", "C", "D", "D"]


def test_combine_structures(example_structure_1):
    lattice = xtal.Lattice(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).transpose()
    )

    atom_coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    ).transpose()

    atom_type = ["A", "B", "B", "B"]

    s1 = xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=atom_coordinate_frac,
        atom_type=atom_type,
    )

    combined = xtal.combine_structures([s1])
    assert combined.atom_type() == atom_type
    print(xtal.pretty_json(combined.to_dict(frac=False)))

    translation = xtal.SymOp(
        matrix=np.eye(3),
        translation=np.array([0.0, 0.0, 3.0]),
        time_reversal=False,
    )
    s2 = translation * s1
    print(xtal.pretty_json(s2.to_dict(frac=False)))
    combined = xtal.combine_structures(
        structures=[s1, s2],
        lattice=xtal.Lattice(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            ).transpose()
        ),
    )
    assert combined.atom_type() == atom_type * 2
