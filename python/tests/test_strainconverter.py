import math

import numpy as np

import libcasm.xtal as xtal

# from scipy.spatial.transform import Rotation


def test_strainconverter_F_to_QU():
    # Qi = Rotation.from_euler('z', 30, degrees=True).as_matrix()
    Qi = np.array([[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]])
    Ui = np.array([[1.0, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.0]])

    Fi = Qi @ Ui
    Q, U = xtal.StrainConverter.F_to_QU(Fi)
    assert np.allclose(Q, Qi)
    assert np.allclose(U, Ui)


def test_strainconverter_F_to_VQ():
    Qi = np.array([[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]])
    Vi = np.array([[1.0, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.0]])

    Fi = Vi @ Qi
    Q, V = xtal.StrainConverter.F_to_VQ(Fi)
    assert np.allclose(Q, Qi)
    assert np.allclose(V, Vi)


def test_strainconverter_Ustrain():
    Qi = np.array([[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]])
    Ui = np.array([[1.0, 0.01, 0.0], [0.01, 1.01, 0.0], [0.0, 0.0, 1.0]])
    w = math.sqrt(2.0)
    U_unrolled = np.array(
        [Ui[0, 0], Ui[1, 1], Ui[2, 2], w * Ui[1, 2], w * Ui[0, 2], w * Ui[0, 1]]
    )
    Fi = Qi @ Ui

    converter = xtal.StrainConverter("Ustrain")

    U_vector = converter.from_F(Fi)
    assert np.allclose(U_vector, U_unrolled)

    F = converter.to_F(U_unrolled)
    assert np.allclose(F, Ui)

    U_matrix = converter.to_E_matrix(U_vector)
    assert np.allclose(U_matrix, Ui)

    U_vector2 = converter.from_E_matrix(U_matrix)
    assert np.allclose(U_vector2, U_vector)


def test_strainconverter_conversions():
    strain_basis = xtal.make_symmetry_adapted_strain_basis()
    assert strain_basis.shape == (6, 6)

    Qi = np.array([[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]])
    Ui = np.array([[1.0, 0.0, 0.0], [0.0, 1.1, 0.2], [0.0, 0.2, 0.9]])
    w = math.sqrt(2.0)
    np.array([Ui[0, 0], Ui[1, 1], Ui[2, 2], w * Ui[1, 2], w * Ui[0, 2], w * Ui[0, 1]])
    Fi = Qi @ Ui

    Q, U = xtal.StrainConverter.F_to_QU(Fi)
    assert np.allclose(Q, Qi)
    assert np.allclose(U, Ui)
    assert np.allclose(Q @ U, Fi)
    Q, Vi = xtal.StrainConverter.F_to_VQ(Fi)
    assert np.allclose(Q, Qi)
    assert np.allclose(Vi @ Q, Fi)

    for metric in ["Ustrain", "Bstrain", "EAstrain", "GLstrain", "Hstrain"]:
        converter = xtal.StrainConverter(metric, strain_basis)

        # round-trip from/to F removes initial rigid rotation, Qi
        e_adapted_basis = converter.from_F(Fi)
        F = converter.to_F(e_adapted_basis)

        if metric == "EAstrain":
            # for EAstrain, remainder is Vi
            Q, V = xtal.StrainConverter.F_to_VQ(F)
            assert np.allclose(Q, np.eye(3))
            assert np.allclose(V, Vi)
        else:
            # for other metrics, remainder is Ui
            Q, U = xtal.StrainConverter.F_to_QU(F)
            assert np.allclose(Q, np.eye(3))
            assert np.allclose(U, Ui)

        # round-trip from/to standard basis results in no change
        e_standard_basis = converter.to_standard_basis(e_adapted_basis)
        e_adapted_basis_2 = converter.from_standard_basis(e_standard_basis)
        assert np.allclose(e_adapted_basis, e_adapted_basis_2)


def test_strainconverter_reduced_dim():
    strain_basis = xtal.make_symmetry_adapted_strain_basis()
    assert strain_basis.shape == (6, 6)
    reduced_strain_basis = xtal.make_symmetry_adapted_strain_basis()[:, :3]
    assert reduced_strain_basis.shape == (6, 3)

    Qi = np.array([[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]])
    Ui = np.array([[1.0, 0.0, 0.0], [0.0, 1.1, 0.1], [0.0, 0.1, 0.9]])
    np.array([[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.9]])
    math.sqrt(2.0)
    Fi = Qi @ Ui

    for metric in ["Ustrain", "Bstrain", "EAstrain", "GLstrain", "Hstrain"]:
        converter = xtal.StrainConverter(metric, strain_basis)
        reduced_converter = xtal.StrainConverter(metric, reduced_strain_basis)

        # round-trip from/to F removes initial rigid rotation, Qi
        e_adapted_basis = converter.from_F(Fi)
        assert e_adapted_basis.shape == (6,)
        reduced_e_adapted_basis = reduced_converter.from_F(Fi)
        assert reduced_e_adapted_basis.shape == (3,)
        assert np.allclose(reduced_e_adapted_basis, e_adapted_basis[:3])

        F = reduced_converter.to_F(reduced_e_adapted_basis)
        Q, V = xtal.StrainConverter.F_to_VQ(F)
        assert np.allclose(Q, np.eye(3))

        # # round-trip from/to standard basis results in no change
        reduced_e_standard_basis = reduced_converter.to_standard_basis(
            reduced_e_adapted_basis
        )
        reduced_e_adapted_basis_2 = reduced_converter.from_standard_basis(
            reduced_e_standard_basis
        )
        assert np.allclose(reduced_e_adapted_basis, reduced_e_adapted_basis_2)
        assert np.allclose(reduced_e_standard_basis[-3:], np.array([0.0, 0.0, 0.0]))
