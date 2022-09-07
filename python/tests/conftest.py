import numpy as np
import libcasm.xtal as xtal
import pytest


@pytest.fixture
def root_pytest_dir(request: pytest.FixtureRequest) -> str:
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


@pytest.fixture
def tetragonal_lattice():

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 1., 0.],  # a
        [0., 0., 2.],  # c
    ]).transpose()
    return xtal.Lattice(lattice_column_vector_matrix)


@pytest.fixture
def simple_cubic_binary_prim():

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 1., 0.],  # a
        [0., 0., 1.],  # a
    ]).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A", "B"]]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     global_dof=global_dof,
                     occupants=occupants)


@pytest.fixture
def simple_cubic_ising_prim():

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 1., 0.],  # a
        [0., 0., 1.],  # a
    ]).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    A_up = xtal.Occupant("A", properties={"Cmagspin": np.array([1.])})
    A_down = xtal.Occupant("A", properties={"Cmagspin": np.array([-1.])})
    occupants = {
        "A.up": A_up,  # A atom, spin up
        "A.down": A_down,  # A atom, spin down
    }
    occ_dof = [
        ["A.up", "A.down"],
    ]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     global_dof=global_dof,
                     occupants=occupants)


@pytest.fixture
def simple_cubic_1d_disp_prim():

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 1., 0.],  # a
        [0., 0., 1.],  # a
    ]).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A"]]

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis(  # Atomic displacement (1d)
        "disp",
        axis_names=["d_{1}"],
        basis=np.array([
            [1.0, 0.0, 0.0],
        ]).transpose())
    local_dof = [[disp_dof]]

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     global_dof=global_dof,
                     occupants=occupants)


@pytest.fixture
def nonprimitive_cubic_occ_prim():

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 2., 0.],  # a
        [0., 0., 1.],  # a
    ]).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],
        [0., 0.5, 0.],
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A", "B"], ["A", "B"]]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     global_dof=global_dof,
                     occupants=occupants)


@pytest.fixture
def perovskite_occ_prim():

    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 1., 0.],  # a
        [0., 0., 1.],  # a
    ]).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [
        ["Sr", "La"],
        ["Ti", "Nb"],
        ["O"],
        ["O"],
        ["O"],
    ]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     global_dof=global_dof,
                     occupants=occupants)


@pytest.fixture
def test_nonprimitive_manydof_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array([
        [1., 0., 0.],  # a
        [0., 2., 0.],  # b
        [0., 0., 1.],  # c
    ]).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array([
        [0., 0., 0.],
        [0., 1.5, 0.],
    ]).transpose()

    # Occupation degrees of freedom (DoF)
    A_up = xtal.Occupant("A", properties={"Cmagspin": np.array([1.])})
    A_down = xtal.Occupant("A", properties={"Cmagspin": np.array([-1.])})
    occupants = {
        "A.up": A_up,  # A atom, spin up
        "A.down": A_down,  # A atom, spin down
    }
    occ_dof = [
        ["A.up", "A.down"],  # site occupants, basis site b=0
        ["A.up", "A.down"],  # site occupants, basis site b=1
    ]

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis("disp")  # Atomic displacement
    local_dof = [
        [disp_dof],  # local DoF, basis site b=0
        [disp_dof],  # local DoF, basis site b=1
    ]

    # Global continuous degrees of freedom (DoF)
    GLstrain_dof = xtal.DoFSetBasis("GLstrain")  # Green-Lagrange strain metric
    global_dof = [GLstrain_dof]

    return xtal.Prim(lattice=lattice,
                     coordinate_frac=coordinate_frac,
                     occ_dof=occ_dof,
                     local_dof=local_dof,
                     global_dof=global_dof,
                     occupants=occupants)
