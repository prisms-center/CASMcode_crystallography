import numpy as np
import pytest

import libcasm.xtal as xtal
import libcasm.xtal.lattices as xtal_lattices


@pytest.fixture
def tetragonal_lattice():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 1.0, 0.0],  # a
            [0.0, 0.0, 2.0],  # c
        ]
    ).transpose()
    return xtal.Lattice(lattice_column_vector_matrix)


@pytest.fixture
def simple_cubic_binary_prim():
    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A", "B"]]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(
        lattice=xtal_lattices.cubic(1.0),
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def simple_cubic_binary_va_disp_Hstrain_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 1.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A", "B", "Va"]]

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis("disp")  # Atomic displacement
    local_dof = [
        [disp_dof],  # local DoF, basis site b=0
    ]

    # Global continuous degrees of freedom (DoF)
    GLstrain_dof = xtal.DoFSetBasis("Hstrain")  # Hencky strain metric
    global_dof = [GLstrain_dof]

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def simple_cubic_ising_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 1.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    A_up = xtal.Occupant("A", properties={"Cmagspin": np.array([1.0])})
    A_down = xtal.Occupant("A", properties={"Cmagspin": np.array([-1.0])})
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

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def simple_cubic_1d_disp_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 1.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A"]]

    # Local continuous degrees of freedom (DoF)
    disp_dof = xtal.DoFSetBasis(  # Atomic displacement (1d)
        "disp",
        axis_names=["d_{1}"],
        basis=np.array(
            [
                [1.0, 0.0, 0.0],
            ]
        ).transpose(),
    )
    local_dof = [[disp_dof]]

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def nonprimitive_cubic_occ_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 2.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A", "B"], ["A", "B"]]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def perovskite_occ_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 1.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    ).transpose()

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

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def test_nonprimitive_manydof_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 2.0, 0.0],  # b
            [0.0, 0.0, 1.0],  # c
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    A_up = xtal.Occupant("A", properties={"Cmagspin": np.array([1.0])})
    A_down = xtal.Occupant("A", properties={"Cmagspin": np.array([-1.0])})
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

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def ZrO_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [3.233986856383, 0.000000000000, 0.000000000000],  # a
            [-1.616993428191, 2.800714773133, 0.000000000000],  # a
            [0.000000000000, 0.000000000000, 5.168678340000],  # c
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 4.0],
            [1.0 / 3.0, 1.0 / 3.0, 3.0 / 4.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["Zr"], ["Zr"], ["O", "Va"], ["O", "Va"]]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def example_structure_1():
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

    # global properties
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

    return xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=xtal.cartesian_to_fractional(
            lattice, atom_coordinate_cart
        ),
        atom_type=["A", "A", "B", "B"],
        atom_properties=atom_properties,
        global_properties=global_properties,
    )


@pytest.fixture
def example_structure_2():
    lattice = xtal.Lattice(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]).transpose()
    )

    atom_coordinate_cart = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 1.5],
        ]
    ).transpose()

    return xtal.Structure(
        lattice=lattice,
        atom_coordinate_frac=xtal.cartesian_to_fractional(
            lattice, atom_coordinate_cart
        ),
        atom_type=["A", "Va", "B", "A"],
    )
