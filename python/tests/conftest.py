import numpy as np
import casm.xtal as xtal
import pytest

@pytest.fixture
def perovskite_occ_prim():

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.], # a
        [0., 1., 0.], # a
        [0., 0., 1.]] # a
        ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]]).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [
        ["Sr", "La"],
        ["Ti", "Nb"],
        ["O"],
        ["O"],
        ["O"]
    ]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(lattice=lattice, coordinate_frac=coordinate_frac, occ_dof=occ_dof,
                     local_dof=local_dof, global_dof=global_dof, occupants=occupants)
