import numpy as np

import libcasm.xtal as xtal


def test_SymOp_constructor():
    op = xtal.SymOp(np.eye(3), np.zeros((3, 1)), False)
    assert np.allclose(op.matrix(), np.eye(3))
    assert np.allclose(op.translation(), np.zeros((3, 1)))
    assert op.time_reversal() is False


def test_SymOp_to_dict():
    op = xtal.SymOp(np.eye(3), np.zeros((3, 1)), False)
    data = op.to_dict()
    assert np.allclose(data["matrix"], op.matrix())
    assert np.allclose(data["tau"], op.translation())
    assert np.allclose(data["time_reversal"], op.time_reversal())


def test_SymOp_from_dict():
    matrix = np.eye(3).tolist()
    translation = [0.0, 0.0, 0.0]
    time_reversal = False
    data = {"matrix": matrix, "tau": translation, "time_reversal": time_reversal}
    op = xtal.SymOp.from_dict(data)
    assert np.allclose(op.matrix(), matrix)
    assert np.allclose(op.translation(), translation)
    assert op.time_reversal() == time_reversal


def test_SymOp_repr():
    import io
    from contextlib import redirect_stdout

    op = xtal.SymOp(np.eye(3), np.zeros((3, 1)), False)

    f = io.StringIO()
    with redirect_stdout(f):
        print(op)
    out = f.getvalue()
    assert "matrix" in out


def test_SymOp_copy():
    import copy

    op = xtal.SymOp(np.eye(3), np.zeros((3, 1)), False)

    op1 = op.copy()
    assert isinstance(op1, xtal.SymOp)
    assert op1 is not op

    op2 = copy.copy(op)
    assert isinstance(op2, xtal.SymOp)
    assert op2 is not op

    op3 = copy.deepcopy(op)
    assert isinstance(op3, xtal.SymOp)
    assert op3 is not op


def test_SymOp_mul_SymOp():
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tau = np.array([1.0, 1.0, 1.0])
    lhs = xtal.SymOp(R, tau, False)

    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tau = np.array([1.0, 1.0, 1.0])
    rhs = xtal.SymOp(R, tau, False)

    combined = lhs * rhs
    r = np.array([1.0, 2.0, 3.0])

    assert np.allclose(combined * r, lhs * (rhs * r))


def test_SymOp_mul_coordinate_2d():
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tau = np.array([1.0, 1.0, 1.0])
    op = xtal.SymOp(R, tau, False)
    r = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    ).transpose()

    r_after = op * r

    r_expected = r.copy()
    for i in range(r.shape[1]):
        r_expected[:, i] = R @ r[:, i] + tau

    assert np.allclose(r_after, r_expected)


def test_SymOp_mul_coordinate_1d():
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tau = np.array([1.0, 1.0, 1.0])
    op = xtal.SymOp(R, tau, False)
    r = np.array([1.0, 2.0, 3.0])

    r_after = op * r
    r_expected = R @ r + tau

    assert np.allclose(r_after, r_expected)


def test_SymOp_mul_properties():
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tau = np.array([1.0, 1.0, 1.0])
    op = xtal.SymOp(R, tau, False)

    # check local DoF 2d array - accepted, returned as (m, n_atoms/n_sites) array
    disp = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.2, 0.3],
        ]
    ).transpose()
    assert disp.shape == (3, 4)
    local_properties = {"disp": disp}
    transformed_properties = op * local_properties
    print(transformed_properties)
    assert "disp" in transformed_properties
    assert transformed_properties["disp"].shape == (3, 4)

    matrix_rep = op.matrix_rep("disp")
    transformed_disp = matrix_rep @ disp
    assert np.allclose(transformed_disp, transformed_properties["disp"])

    # check 1d array - accepted, but returned as (6,1) array
    Hstrain = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert Hstrain.shape == (6,)
    print("Hstrain.shape:", Hstrain.shape)
    global_properties = {"Hstrain": Hstrain}
    assert global_properties["Hstrain"].shape == (6,)
    transformed_properties = op * global_properties
    print(transformed_properties)
    assert "Hstrain" in transformed_properties
    assert transformed_properties["Hstrain"].shape == (6, 1)

    matrix_rep = op.matrix_rep("Hstrain")
    transformed_Hstrain = matrix_rep @ Hstrain.reshape(-1, 1)
    assert np.allclose(transformed_Hstrain, transformed_properties["Hstrain"])

    # check 2d column array - accepted, stays as (6,1) array
    Hstrain = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]).transpose()
    assert Hstrain.shape == (6, 1)
    print("Hstrain.shape:", Hstrain.shape)
    global_properties = {"Hstrain": Hstrain}
    assert global_properties["Hstrain"].shape == (6, 1)
    transformed_properties = op * global_properties
    print(transformed_properties)
    assert "Hstrain" in transformed_properties
    assert transformed_properties["Hstrain"].shape == (6, 1)

    matrix_rep = op.matrix_rep("Hstrain")
    transformed_Hstrain = matrix_rep @ Hstrain
    assert np.allclose(transformed_Hstrain, transformed_properties["Hstrain"])


def test_SymOp_mul_lattice(tetragonal_lattice):
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tau = np.array([1.0, 1.0, 1.0])
    op = xtal.SymOp(R, tau, False)

    transformed_lattice = op * tetragonal_lattice

    expected_L = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
            [0.0, 2.0, 0.0],  # c
        ]
    ).transpose()

    assert np.allclose(transformed_lattice.column_vector_matrix(), expected_L)


def test_SymOp_mul_structure(example_structure_1):
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tau = np.array([1.0, 1.0, 1.0])
    op = xtal.SymOp(R, tau, False)

    structure = example_structure_1
    transformed_structure = op * structure

    expected_L = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
            [0.0, 2.0, 0.0],  # c
        ]
    ).transpose()
    expected_atom_coordinate_cart = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 1.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    ).transpose()
    expected_disp = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.1, 0.0],
            [0.1, 0.3, 0.2],
        ]
    ).transpose()
    expected_Hstrain = np.array(
        [[0.009950330853168087, 0.0, 0.0, 0.0, 0.0, 0.0]]
    ).transpose()

    assert np.allclose(
        transformed_structure.lattice().column_vector_matrix(), expected_L
    )
    print(structure.atom_coordinate_cart())
    print(transformed_structure.atom_coordinate_cart())
    print(expected_atom_coordinate_cart)
    assert np.allclose(
        transformed_structure.atom_coordinate_cart(), expected_atom_coordinate_cart
    )
    assert np.allclose(transformed_structure.atom_properties()["disp"], expected_disp)
    assert np.allclose(
        transformed_structure.global_properties()["Hstrain"], expected_Hstrain
    )
