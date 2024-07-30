"""Contains functions related to finding the point group of a cluster of atoms or a molecule"""

import itertools
import numpy as np
import libcasm.xtal as xtal
from typing import List, Tuple, Union


def geometric_center_of_mass(molecule: xtal.Occupant) -> np.ndarray:
    """Calculates the geometric center of mass of the molecule

    :math:`\\mathbf{N} \\rightarrow` Number of atoms in molecule \n
    :math:`\\mathbf{r}_i \\rightarrow` coordinates of :math:`i_{\\mathbf{th}}` atom \n
    :math:`\\mathbf{r}_{com} \\rightarrow` coordinates of geometric center of mass

    .. math::
        \\mathbf{r}_{com} = \\frac{\\sum^{\\mathbf{N}}_{i=1} \\mathbf{r}_i}{\\mathbf{N}}

    Parameters
    ----------
    molecule : xtal.Occupant
        Molecule for which you want the center of mass

    Returns
    -------
    numpy.ndarray
        Coordinates of geometric center of mass

    """
    cart_coords = [atom.coordinate() for atom in molecule.atoms()]
    geometric_center_of_mass = np.zeros(cart_coords[0].shape)

    for coord in cart_coords:
        geometric_center_of_mass += coord

    return geometric_center_of_mass / len(cart_coords)


def shift_origin_of_molecule_to_geometric_center_of_mass(
    molecule: xtal.Occupant,
) -> xtal.Occupant:
    """Shifts the molecule to geometric center of mass

    :math:`\\mathbf{N} \\rightarrow` Group of all atoms in a molecule \n
    :math:`\\mathbf{r}_i \\rightarrow` coordinates of :math:`i_{\\mathbf{th}}` atom \n
    :math:`\\mathbf{r}_{com}` coordinates of geometric center of mass

    .. math::
        \\Big\\{ \\mathbf{r}_i - \\mathbf{r}_{com} \\Big| \\; \\forall \\; i  \\; \\in \\; \\mathbf{N} \\Big\\}

    Parameters
    ----------
    molecule : xtal.Occupant
        Molecule for which you want shift the origin to it's geometric center of mass

    Returns
    -------
    xtal.Occupant
        Molecule where the origin is it's geometric center of mass

    """
    center_of_mass = geometric_center_of_mass(molecule)
    new_atoms = [
        xtal.AtomComponent(
            name=atom.name(),
            coordinate=(atom.coordinate() - center_of_mass),
            properties=atom.properties(),
        )
        for atom in molecule.atoms()
    ]

    return xtal.Occupant(name=molecule.name(), atoms=new_atoms)


def projection_operator(array: np.ndarray) -> np.ndarray:
    """Make a projection operator for a given numpy array

    .. math::
        \\mathbf{R}_{proj} =  \\mathbf{R} \\mathbf{R}^{T}

    Parameters
    ----------
    array : numpy.ndarray
        Numpy vector for which you want to compute projection operator

    Returns
    -------
    numpy.ndarray
        Projection operator

    """

    if len(array.shape) == 1:
        array = np.transpose(array[np.newaxis])

    return array @ np.transpose(array)


def is_molecule_planar(molecule: xtal.Occupant, tol: float = 1e-5) -> bool:
    """Checks if the provided molecule is planar

    :math:`\\mathbf{N} \\rightarrow` Group of all atoms in a molecule \n
    :math:`\\mathbf{r}_i \\rightarrow` coordinates of :math:`i_{\\mathbf{th}}` atom \n
    :math:`\\mathbf{r}^i_{proj} \\rightarrow` Projection operator of :math:`i_{\\mathbf{th}}` atom
    (:math:`\\mathbf{r}_i \\mathbf{r}^{T}`)\n

    If only one of the eigen values of :math:`\\sum^{\\mathbf{N}}_{i=1} \\mathbf{r}^{i}_{proj}`
    is zero, then the molecule is planar

    Parameters
    ----------
    molecule : .core.Structure.Molecule
        Molecule for which you want to compute it's planarity
    tol : float, optional
        Tolerance used for comparisions.

    Returns
    -------
    bool
        Whether the molecule is planar or not
    """
    molecule_shifted_to_geometric_center_of_mass = (
        shift_origin_of_molecule_to_geometric_center_of_mass(molecule)
    )

    summed_projection_opertors_of_coords = projection_operators_sum_of_list_of_arrays(
        get_coord_list_from_molecule(molecule_shifted_to_geometric_center_of_mass)
    )

    eigen_vals, _ = np.linalg.eig(summed_projection_opertors_of_coords)

    if (
        len(
            [
                eigen_val
                for eigen_val in eigen_vals
                if np.allclose(eigen_val, 0, tol, tol)
            ]
        )
        == 1
    ):
        return True

    return False


def is_molecule_linear(molecule: xtal.Occupant, tol: float = 1e-5) -> bool:
    """Checks if the provided molecule is linear

    :math:`\\mathbf{N} \\rightarrow` Group of all atoms in a molecule \n
    :math:`\\mathbf{r}_i \\rightarrow` coordinates of :math:`i_{\\mathbf{th}}` atom \n
    :math:`\\mathbf{r}^i_{proj} \\rightarrow` Projection operator of :math:`i_{\\mathbf{th}}` atom
    (:math:`\\mathbf{r}_i \\mathbf{r}^{T}`)\n

    If only two of the eigen values of :math:`\\sum^{\\mathbf{N}}_{i=1} \\mathbf{r}^{i}_{proj}`
    is zero, then the molecule is linear

    Parameters
    ----------
    molecule : xtal.Occupant
        Molecule for which you want to compute it's linearity
    tol : float, optional
        Tolerance used for comparisions

    Returns
    -------
    bool
        Whether the molecule is linear or not
    """

    molecule_shifted_to_geometric_center_of_mass = (
        shift_origin_of_molecule_to_geometric_center_of_mass(molecule)
    )

    summed_projection_opertors_of_coords = projection_operators_sum_of_list_of_arrays(
        get_coord_list_from_molecule(molecule_shifted_to_geometric_center_of_mass)
    )

    eigen_vals, _ = np.linalg.eig(summed_projection_opertors_of_coords)

    if (
        len(
            [
                eigen_val
                for eigen_val in eigen_vals
                if np.allclose(eigen_val, 0, tol, tol)
            ]
        )
        == 2
    ):
        return True

    return False


def projection_operators_sum_of_list_of_arrays(
    array_list: List[np.ndarray],
) -> np.ndarray:
    """For a given list of arrays, compute the projection operator of each array
    and compute it's sum

    :math:`\\mathbf{N} \\rightarrow` Total number of arrays in the given list \n
    :math:`\\mathbf{R}_i \\rightarrow` :math:`i_{\\mathbf{th}}` array in the list \n

    This function returns

    .. math::
        \\sum^{\\mathbf{N}}_{i=1} \\mathbf{R} \\mathbf{R}^{T}

    Parameters
    ----------
    array_list : List[numpy.ndarray]
        List of arrays

    Returns
    -------
    numpy.ndarray
        Summation of projection operators of each array in the given list

    """
    summed_projection_operators_array = np.zeros(
        (array_list[0].shape[0], array_list[0].shape[0])
    )

    for array in array_list:
        summed_projection_operators_array += projection_operator(array)

    return summed_projection_operators_array


def check_if_two_molecules_are_equivalent(
    molecule1: List[Tuple[str, np.ndarray]],
    molecule2: List[Tuple[str, np.ndarray]],
    tol: float = 1e-5,
) -> bool:
    """Checks if the given molecules are equivalent
    The molecules are provided as a list of tuples where each tuple represents
    an atom type and it's corresponding cartesian coordinates.\n
    Compares the given molecules by checking if each atom of a molecule 1 is also
    present in molecule 2.

    Parameters
    ----------
    molecule1 : List[Tuple[str, numpy.ndarray]]
        Molecule 1
    molecule2 : List[Tuple[str, numpy.ndarray]]
        Molecule 2
    tol : float, optional
        Tolerance used for comparisions

    Returns
    -------
    bool
        Whether the given molecules are equivalent or not

    """
    if len(molecule1) == len(molecule2):
        for atom1 in molecule1:
            if (
                len(
                    [
                        True
                        for atom2 in molecule2
                        if (atom1[0] == atom2[0])
                        and np.allclose(atom1[1], atom2[1], tol, tol)
                    ]
                )
                == 0
            ):
                return False

        return True

    return False


def apply_transformation_matrix_to_molecule(
    transformation_matrix: np.ndarray, molecule: List[Tuple[str, np.ndarray]]
) -> List[Tuple[str, np.ndarray]]:
    """Applies the given transformation matrix to the molecule

    :math:`\\mathbf{N} \\rightarrow` Group of all atoms in the molecule \n
    :math:`\\mathbf{T} \\rightarrow` Transformation matrix \n
    :math:`\\mathbf{r}_i \\rightarrow` Cartesian coordinates of :math:`i_{\\mathbf{th}}` atom in the molecule \n
    :math:`el_i \\rightarrow` Element name of :math:`i_{\\mathbf{th}}` atom in the molecule \n

    This function returns

    .. math::
        \\Big \\{ ( el_i, \\mathbf{T} \\, \\mathbf{r}_i ) \\Big| \\; \\forall \\; i \\; \\in \\; \\mathbf{N}  \\Big \\}

    Parameters
    ----------
    transformation_matrix : numpy.ndarray
        The number of columns of the transformation matrix should match
        the dimensions of the coordinate space
    molecule : List[Tuple[str, numpy.ndarray]]
        Molecule to which you want to apply the given transformation matrix

    Returns
    -------
    List[Tuple[str, numpy.ndarray]]
        Transformed molecule

    Raises
    ------
    ValueError
        If number of columns of the ``transformation_matrix`` don't match
        with the dimensions of the coordinate space

    """
    if (
        transformation_matrix.shape[len(transformation_matrix.shape) - 1]
        != molecule[0][1].shape[0]
    ):
        raise ValueError(
            "Columns of the transformation matrix do not match with the dimensions of the coordinate space"
        )

    return [(element, transformation_matrix @ coord) for element, coord in molecule]


def check_if_transformation_matrix_is_symmetry_opertion(
    transformation_matrix: np.ndarray,
    molecule: List[Tuple[str, np.ndarray]],
    tol: float = 1e-5,
) -> bool:
    """Checks if the transformation matrix you obtained in the process
    of making the point group is a symmetry opertaion of the provided molecule.

    :math:`\\mathbf{T} \\rightarrow` Transformation matrix\n
    The given transformation matrix is a symmetry operation under the following conditions:\n
        * :math:`\\mathbf{T}` should be an orthogonal matrix (:math:`\\mathbf{T} \\mathbf{T}^T = \\mathbf{I}`)
        * Applying :math:`\\mathbf{T}` to the given molecule should result in an equivalent molecule

    Parameters
    ----------
    transformation_matrix : numpy.ndarray
        Transformation matrix
    molecule : List[Tuple[str, numpy.ndarray]]
        Molecule represented as list of tuple of element name and it's corresponding cartesian
        coordinate
    tol : float, optional
        Tolerance used for matrix comparisions

    Returns
    -------
    bool
        Whether the given transformation matrix is a symmetry operation

    """

    # check for orthogonility
    if np.allclose(
        projection_operator(transformation_matrix),
        np.identity(transformation_matrix.shape[0]),
        tol,
        tol,
    ):

        # check whether the transformation matrix transforms the molecule into another equivalent molecule
        transformed_molecule = apply_transformation_matrix_to_molecule(
            transformation_matrix, molecule
        )

        if check_if_two_molecules_are_equivalent(molecule, transformed_molecule, tol):
            return True

    return False


def convert_list_of_atom_coords_to_numpy_array(coords: List[np.ndarray]) -> np.ndarray:
    """Converts a given list of coordinates into a numpy array where each
    column is atom coordinates

    Parameters
    ----------
    coords : List[numpy.ndarray]
        List of coordinates that you want to convert to a numpy array

    Returns
    -------
    numpy.ndarray
        A matrix where columns of the matrix are coordinates of atoms

    """
    return np.transpose(np.array(coords))


def get_coords_of_a_given_type_in_molecule(
    molecule: xtal.Occupant, atom_type: str
) -> List[np.ndarray]:
    """Given a molecule, return a list of cartesian coordinates of molecule for a specified
    atom type

    Parameters
    ----------
    molecule : xtal.Occupant
        Molecule
    atom_type : str
        Atom type for which you want to get the cartesian coordinates

    Returns
    -------
    List[numpy.ndarray]
        List of cartesian coordinates of atoms in a given molecule belonging to a specified type

    """
    return [atom.coordinate() for atom in molecule.atoms() if atom.name() == atom_type]


def get_type_of_atoms_in_a_molecule(molecule: xtal.Occupant) -> List[str]:
    """Returns all the unique atom types in a given molecule

    Parameters
    ---------
    molecule: xtal.Occupant
        Molecule

    Returns
    -------
    List[str]
        List of all atom types in a molecule (Does not contain repeats)

    """

    return list(set([atom.name() for atom in molecule.atoms()]))


def get_list_of_atom_type_and_coord_from_molecule(
    molecule: xtal.Occupant,
) -> List[Tuple[str, np.ndarray]]:
    """Given a molecule, return a list of tuples where each tuple contains an
    atom type and it's cartesian coordinates

    Parameters
    ----------
    molecule : xtal.Occupant
        Molecule

    Returns
    -------
    List[Tuple[str, numpy.ndarray]]
        List of tuples where each tuple contains atom type and it's corresponding cartesian coordinates

    """
    return [(atom.name(), atom.coordinate()) for atom in molecule.atoms()]


def get_coord_list_from_molecule(molecule: xtal.Occupant) -> List[np.ndarray]:
    """Get list of cartesian coordinates of a given molecule

    Parameters
    ----------
    molecule : xtal.Occupant
        Molecule

    Returns
    -------
    List[numpy.ndarray]
        List of all cartesian coordinates of a given molecule

    """
    return [atom.coordinate() for atom in molecule.atoms()]


def check_if_the_molecule_has_inversion_symmetry(
    molecule: xtal.Occupant, tol: float = 1e-5
) -> bool:
    """
    Checks if the given molecule has inversion symmetry

    Parameters
    ----------
    molecule : Molecule
        Molecule
    tol : float, optional
        Tolerance used for comparisions

    Returns
    -------
    bool
        True if molecule as inversion symmetry, else False

    """

    molecule = get_list_of_atom_type_and_coord_from_molecule(molecule)
    transformation_matrix = np.identity(molecule[0][1].shape[0]) * -1
    molecule_after_inveresion = apply_transformation_matrix_to_molecule(
        transformation_matrix, molecule
    )

    return check_if_two_molecules_are_equivalent(
        molecule, molecule_after_inveresion, tol
    )


def is_symop_unique(sym_op: np.ndarray, sym_ops: List[np.ndarray], tol: float):
    """Checks if the given sym_op is contained in the list of sym_ops

    Parameters
    ----------
    sym_op : numpy.ndarray
        sym_op that needs to be checked whether it is present in sym_ops
    sym_ops : List[numpy.ndarray]
        List of sym_ops in which whether the given sym_op is present or not
    tol : float
        Tolerance used for matrix comparisions

    Returns
    -------
    bool
        True if sym_op exists in sym_ops, else False

    """
    for s in sym_ops:
        if np.allclose(sym_op, s, tol, tol):
            return False
    return True


def get_initial_combination(
    number_of_atoms_of_a_given_type: int, is_planar: bool
) -> List[int]:
    """
    Not intended for public use. Only to make the code more readable.
    Get an initial combination of integers which can be used a pivoting point to find various
    transformation matrices.

    The initial combination of integers depends on different conditions:\n
        * If the molecule is planar:\n
            * initial combination is [0, 1]
        * If the molecule is non-planar:\n
            * initial combination is [0, 1] if the number of atoms of a given type is 2
            * initial combination is [0, 1, 2] if the number of atoms of a given type is greater than 2

    Parameters
    ----------
    number_of_atoms_of_a_given_type : int
        Number of atoms in a molecule with a specified type
    is_planar : bool
        Planarity of molecule

    Returns
    -------
    List[int]
        Initial combination of integers

    Raises
    ------
    RuntimeError
        If number of atoms is less than zero it happens

    """
    if number_of_atoms_of_a_given_type == 2 and is_planar is False:
        initial_combination = list(range(2))
    elif number_of_atoms_of_a_given_type > 2 and is_planar is False:
        initial_combination = list(range(3))
    elif number_of_atoms_of_a_given_type >= 2 and is_planar is True:
        initial_combination = list(range(2))
    else:
        raise ValueError(
            "You should not have come here. If you did, please know that I am not programmed"
            "to handle cases where number of atoms is less than or equal to zero."
        )
    return initial_combination


def get_all_permutations(
    number_of_atoms_of_a_given_type: int, is_planar: bool
) -> List[int]:
    """
    Not intended for public use. Only to make the code more readable.
    Get all permutations of integers which can be used to find various transformation matrices.

    :math:`\\mathbf{L} \\rightarrow` List of all integers - [0, ``number_of_atoms_of_a_given_type``]\n
    This function returns:\n
        * If the molecule is planar:\n
            * All permutations and combinations of integers of dimensions 2 ([:math:`x`, :math:`y`]) from :math:`\\mathbf{L}`
        * If the molecule is non-planar:\n
            * If the ``number_of_atoms_of_a_given_type`` is 2, all permutations and combinations of integers of dimensions 2 [:math:`x`, :math:`y`] from :math:`\\mathbf{L}`\n
            * If the ``number_of_atoms_of_a_given_type`` is greater than 2, all permutations and combinations of integers of dimensions 3 [:math:`x`, :math:`y`, :math:`z`] from :math:`\\mathbf{L}`

    Parameters
    ----------
    number_of_atoms_of_a_given_type : int
        Number of atoms in a molecule with a specified type
    is_planar : bool
        Planarity of molecule

    Returns
    -------
    List[int]
        List of all permutations

    Raises
    ------
    RuntimeError
        If number of atoms is less than zero it happens
    """
    if number_of_atoms_of_a_given_type == 2 and is_planar is False:
        all_permutations = list(
            itertools.permutations(range(number_of_atoms_of_a_given_type), 2)
        )
    elif number_of_atoms_of_a_given_type > 2 and is_planar is False:
        all_permutations = list(
            itertools.permutations(range(number_of_atoms_of_a_given_type), 3)
        )

    elif number_of_atoms_of_a_given_type >= 2 and is_planar is True:
        all_permutations = list(
            itertools.permutations(range(number_of_atoms_of_a_given_type), 2)
        )
    else:
        raise RuntimeError(
            "You should not have come here. If you did, please know that I am not programmed"
            "to handle cases where number of atoms is less than or equal to zero."
        )
    return all_permutations


def get_transformation_matrices(
    initial_coords: np.ndarray,
    all_permuted_coords: List[np.ndarray],
    number_of_atoms_of_a_given_type: int,
    is_planar: bool,
) -> List[np.ndarray]:
    """
    Not intended for public use. Only to make the code more readable.
    Returns transformation matrices which might be symmetry operations. All these matrices
    are generated from permuting atoms of same type.

    :math:`\\mathbf{r}_{init} \\rightarrow` A numpy matrix where columns of the matrix are cartesian coordinates
    of atoms corresponding to an initial combination of atoms\n
    :math:`\\mathbf{r}_{i} \\rightarrow` A numpy matrix where columns of the matrix are cartesian coordinates of atoms
    corresponding to :math:`i_{\\mathbf{th}}` permutation\n
    :math:`n \\rightarrow` Group of all permutations\n
    :math:`\\mathbf{L} \\rightarrow` List of all transformation matrices\n

    If number of atoms of a given type is 2 and the molecule is non-planar:\n

    .. math::
        L = \\Big\\{ ( \\mathbf{r}_{i} \\mathbf{r}_{init}^{T})(\\mathbf{r}_{init} \\mathbf{r}_{init} ^{T})^{-1}
        \\Big| \\; \\forall \\; i \\in \\; n  \\Big\\}

    If number of atoms of a given type is greater than 2 and the molecule is non-planar or if the number
    of atoms of a given type is greater than equal to 2 and the molecule is planar:\n

    .. math::
        L = \\Big\\{ \\mathbf{r}_{i} \\mathbf{r}_{init}^{-1}
        \\Big| \\; \\forall \\; i \\in \\; n  \\Big\\}

    Parameters
    ----------
    initial_coords : numpy.ndarray
        A numpy matrix where columns of the matrix are cartesian coordinates of atoms corresponding to an initial combination
    all_permuted_coords : List[numpy.ndarray]
        List of all matrices where columns of the matrix are cartesian coordinates of atoms corresponding to all permutations
        and combinations
    number_of_atoms_of_a_given_type : int
        Number of atoms in a molecule of a specified type
    is_planar : bool
        Planarity of the molecule

    Returns
    -------
    List[numpy.ndarray]
        List of transformation matrices

    """
    transformation_matrices = [] * len(all_permuted_coords)
    # If number of atoms of a given type is 2, C is not a square matrix
    # T = (C_prime * transpose(C)) * inverse(C * transpose(C)) = (L_prime) * L_matrix
    if number_of_atoms_of_a_given_type == 2 and is_planar is False:
        l_matrix = np.linalg.inv(initial_coords @ np.transpose(initial_coords))
        for c_prime in all_permuted_coords:
            l_prime = c_prime @ np.transpose(initial_coords)
            t = l_prime @ l_matrix
            transformation_matrices.append(t)

    # If number of atoms of a given type is > 2,
    # T = C_prime * inverse(C)
    if (number_of_atoms_of_a_given_type > 2 and is_planar is False) or (
        number_of_atoms_of_a_given_type >= 2 and is_planar is True
    ):
        for c_prime in all_permuted_coords:
            t = c_prime @ np.linalg.inv(initial_coords)
            transformation_matrices.append(t)

    return transformation_matrices


def make_symops_from_same_atom_type_coords(
    atom_coords_of_a_given_type: List[np.ndarray],
    molecule: List[Tuple[str, np.ndarray]],
    is_planar: bool,
    tol: float,
) -> List[np.ndarray]:
    """
    Not intended for public use. Only to make the code more readable.
    Makes symmetry operations from coordinates of same type of atoms.

    If the molecule is planar ``atom_coords_of_a_given_type`` should be reduced
    dimensions (:math:`2 \\times 1``) and it should be in a coordinate space
    where the planar molecule is always perpendicular to the :math:`z` axis. \n

    If the molecule is planar ``molecule`` should be in a coordinate space where
    it is always perpendicular to the :math:`z` axis. \n

    To acheive these two criterion please use ``transformation_matrix_to_eigen_vector_space`` and
    ``transform_coords_into_evs_for_planar_molecule`` helper functions.

    Parameters
    ----------
    atom_coords_of_a_given_type : List[numpy.ndarray]
    molecule : List[Tuple[str, numpy.ndarray]]
        Molecule. Only used for checking if the obtained transformation matrix is a symmetry operation
    is_planar : bool
        Planarity of molecule
    tol : float
        Tolerance used for comparisons

    Returns
    -------
    List[numpy.ndarray]
        List of symmetry operations

    """

    number_of_atoms_of_a_given_type = len(atom_coords_of_a_given_type)
    sym_ops = []

    # If you have only one atom of a given type
    # All you can deduce is that Identity will be a sym op
    if number_of_atoms_of_a_given_type == 1:
        identity_symop = np.identity(molecule[0][1].shape[0])
        if is_symop_unique(identity_symop, sym_ops, tol):
            sym_ops.append(identity_symop)
        return sym_ops

    # Make an initial (3 x n) coordinate matrix where columns of the matrix are coords of atoms
    initial_combination = get_initial_combination(
        number_of_atoms_of_a_given_type, is_planar
    )
    initial_coords = convert_list_of_atom_coords_to_numpy_array(
        atom_coords_of_a_given_type
    )[:, initial_combination]

    # Now get all the possible permutations and combinations of (3xn) matrices
    # where columns of the matrices are cartesian coordinates atoms of a given type
    all_permutations = get_all_permutations(number_of_atoms_of_a_given_type, is_planar)
    all_permuted_coords = [
        convert_list_of_atom_coords_to_numpy_array(atom_coords_of_a_given_type)[
            :, permutation
        ]
        for permutation in all_permutations
    ]

    # Find a transformation_matrices which takes initial coordinate matrix to permuted coordinates
    # matrix. These are possible symmetry operations.
    possible_sym_ops = get_transformation_matrices(
        initial_coords,
        all_permuted_coords,
        number_of_atoms_of_a_given_type,
        is_planar,
    )

    # Now these transformation matrices can only be symmetry operations of the entire
    # molecule if and only if the molecule remains unchanged when you apply this transformation matrix
    # to the whole molecule
    if is_planar is False:
        for possible_op in possible_sym_ops:

            if check_if_transformation_matrix_is_symmetry_opertion(
                possible_op, molecule, tol
            ) and is_symop_unique(possible_op, sym_ops, tol):
                sym_ops.append(possible_op)

    if is_planar is True:
        # If the molecule is planar, you end up getting 2x2 transformation matrices, since you are
        # working in a reduced dimensions. Now you convert these 2x2 matrices into 3x3 matrices by
        # adding another eigen vector +z or -z to it, since you are working in a space where the
        # molecule is always perpendicular to z axis (this you have to ensure while passing molecule
        # argument to the function). You also have to ensure that atom_coords_of_a_given_type will
        # be 2 dimensional if your molecule is known to be planar.
        new_possible_sym_ops = []
        for possible_op in possible_sym_ops:
            new_possible_sym_ops.append(
                np.block([[possible_op[0, :], 0], [possible_op[1, :], 0], [0, 0, 1]])
            )
            new_possible_sym_ops.append(
                np.block([[possible_op[0, :], 0], [possible_op[1, :], 0], [0, 0, -1]])
            )

        for op in new_possible_sym_ops:
            if check_if_transformation_matrix_is_symmetry_opertion(
                op, molecule, tol
            ) and is_symop_unique(op, sym_ops, tol):
                sym_ops.append(op)

    return sym_ops


def transformation_matrix_to_eigen_vector_space(
    molecule: xtal.Occupant, tol: float
) -> np.ndarray:
    """
    Transformation matrix that transforms the given planar molecule into eigen vector space
    such that it is perpendicular to :math:`z` axis.

    Parameters
    ----------
    molecule: xtal.Occupant
        Planar molecule
    tol : float
        Tolerance used for comparisons

    Returns
    -------
    numpy.ndarray
        A matrix which transforms the planar molecule into eigen vector space such that it is
        perpendicular to :math:`z` axis.

    """
    summed_projection_operators = projection_operators_sum_of_list_of_arrays(
        get_coord_list_from_molecule(molecule)
    )

    eigen_vals, eigen_vector_space = np.linalg.eig(summed_projection_operators)

    index_corresponding_to_zero_eigen_val = [
        i
        for i, eigen_val in enumerate(eigen_vals)
        if np.allclose(eigen_val, 0, tol, tol)
    ][0]

    coords_with_atom_types = get_list_of_atom_type_and_coord_from_molecule(molecule)

    # And now rotate the molecule in eigen vector space, so that the z coordinates of atom
    # is always zero
    identity = np.identity(coords_with_atom_types[0][1].shape[0])
    permutation = (
        list(range(0, index_corresponding_to_zero_eigen_val))
        + list(
            range(
                index_corresponding_to_zero_eigen_val + 1,
                coords_with_atom_types[0][1].shape[0],
            )
        )
        + [index_corresponding_to_zero_eigen_val]
    )
    rotation_matrix = np.transpose(identity[:, permutation])

    return rotation_matrix @ np.transpose(eigen_vector_space)


def transform_coords_into_reduced_evs_for_planar_molecule(
    atom_type: str,
    atom_coords_of_a_given_type: List[np.ndarray],
    transf_mat_to_evs: np.ndarray,
) -> List[np.ndarray]:
    """
    Not intended for public use. Only to make the code more readable.
    This particular function gives atom coordinates for a given atom type
    in a reduced (:math:`2 \\times 1`) dimensions in a eigen vector space
    where the planar molecule is perpendicular to :math:`z` axis.

    Should only be used when the molecule is planar

    Parameters
    ----------
    atom_type : str
        Atom type
    atom_coords_of_a_given_type : List[numpy.ndarray]
        Atom coordinates of a given atom type in cartesian space
    transf_mat_to_evs : numpy.ndarray
        Transformation matrix which transforms the molecule into
        eigen vector space where the planar molecule is perpendicular to :math:`z` axis

    Returns
    -------
    List[numpy.ndarray]
        Reduced dimension coordinates of atoms in eigen vector space where
        the planar molecule is perpendicular to :math:`z` axis.

    """
    atom_coords_of_a_given_type_with_atom_type = [
        (atom_type, coord) for coord in atom_coords_of_a_given_type
    ]

    atom_coords_of_a_given_type_in_evs_with_z_nulled = (
        apply_transformation_matrix_to_molecule(
            transf_mat_to_evs,
            atom_coords_of_a_given_type_with_atom_type,
        )
    )
    atom_coords_of_a_given_type_in_evs_with_z_axed = [
        coord[1][0:-1] for coord in atom_coords_of_a_given_type_in_evs_with_z_nulled
    ]

    return atom_coords_of_a_given_type_in_evs_with_z_axed


def issue_warning_if_origin_is_not_center_of_geometry(
    molecule: xtal.Occupant, tol: float
) -> None:
    """If center of geometry of a molecule is not at origin,
    issue a warning

    Parameters
    ----------
    molecule : Molecule
        Molecule that you are interested in
    tol : float
        Tolerance for comparisions

    Returns
    -------
    None

    """
    if not np.allclose(
        geometric_center_of_mass(molecule),
        np.zeros(molecule.atoms()[0].coordinate().shape),
        tol,
        tol,
    ):
        print(
            "WARNING: Geometric center of mass of the molecule is not at origin. Shifting it to origin. If you are working on the molecule further please shift it's geometric center of mass to origin."
        )
    return None


def cluster_point_group(
    molecule: xtal.Occupant, tol: float = 1e-5
) -> Union[str, List[np.ndarray]]:
    """Computes the point group of a cluster of atoms or a molecule

    The algorithm to compute the point group of molecule is divided into various parts:\n

    1. If the molecule is linear, there can be infinite symmetry operations possible in the direction of
    long axis, hence this function only returns the name of the point group.\n

    2. If the molecule is planar,\n
    The molecule is first transformed into eigen vector space and also will be made sure it is always
    perpendicular to the :math:`z` axis. All the coordinates of atoms are represented in a reduced dimensions (:math:`2 \\times 1`)
    in this vector space. Now for a particular atom type, you pick two atom coordinates each of dimension :math:`2\\times1`,
    and make an initial :math:`2 \\times 2` (let's call it :math:`\\mathbf{R}`) matrix where
    the columns of this matrix are coordinates of the two atoms. Now you make multiple other similar :math:`2\\times2` matrices by
    looking at all the possible permutations and combinations of coordinates of atoms for the atom type you picked initially (
    let's call one of these matrices :math:`\\mathbf{R}^\\prime`). Now you compute various possible transformation matrices
    by :math:`\\mathbf{R}^\\prime \\mathbf{R}^{-1}`. Now the dimensions of these transformation matrices are extended
    to :math:`3 \\times 3` by adding the third axis :math:`\\pm z`. You repeat this process for multiple atom types and collect
    all the transformation matrices. Now you filter symmetry operations from it by checking if the transformation
    matrices are orthogonal and that the molecule is invariant to it. If either of these two criterion are not met,
    the transformation matrix is discarded. Now these symmetry operations are transformed back into cartesian vector space.

    3. If the molecule is non-planar,\n
    a similar procedure as outlined for planar molecule is followed except you work in the normal cartesian space and all
    the coordinates of atoms are :math:`3\\times1` vectors and the transformation matrices are :math:`3 \\times 3` matrices.

    Parameters
    ----------
    molecule : xtal.Occupant
        cluster of atoms for which you want the point group
    tol : float, optional
        Tolerance used for comparisons. The results will be
        very sensitive if the given molecule has lot of
        numerical noise.

    Returns
    -------
    Union[str, List[numpy.ndarray]]
        List of all 3x3 orthornormal matrices that form
        the point group of a given molecule or a string if the
        molecule is linear

    """

    # Shift the molecule to geometric center of mass
    issue_warning_if_origin_is_not_center_of_geometry(molecule, tol)
    molecule = shift_origin_of_molecule_to_geometric_center_of_mass(molecule)

    # check if the molecular is linear
    if is_molecule_linear(molecule, tol):

        # If the molecule has inversion symmetry and linear it belongs to D-inf Eh group
        if check_if_the_molecule_has_inversion_symmetry(molecule, tol):
            raise NotImplementedError(
                "Method not implemented to get point group of a linear molecule"
            )
            # return "D\u221Eh"

        # If the molecule has no inversion symmetry and linear it belongs to C-inf Ev group
        raise NotImplementedError(
            "Method not implemented to get point group of a linear molecule"
        )
        # return "C\u221Ev"

    point_group = []
    # Identity is always a sym op
    point_group.append(np.identity(molecule.atoms()[0].coordinate().shape[0]))

    # Get different atom types in the molecule
    atom_types = get_type_of_atoms_in_a_molecule(molecule)
    # Get molecule as a list of tuple of atom types and their corresponding cartesian coordinates
    atom_coords_along_with_atom_types = get_list_of_atom_type_and_coord_from_molecule(
        molecule
    )

    # If the molecular is planar and not linear
    if (
        is_molecule_planar(molecule, tol) is True
        and is_molecule_linear(molecule, tol) is False
    ):

        # Transformation matrix that makes z coordinate of planar molecule zero
        # You work in the eigen vector space where the molecule is perpendicular to the z axis
        transf_mat_to_evs = transformation_matrix_to_eigen_vector_space(molecule, tol)
        atom_coords_with_atom_types_in_evs_with_z_nulled = (
            apply_transformation_matrix_to_molecule(
                transf_mat_to_evs,
                atom_coords_along_with_atom_types,
            )
        )

        for atom_type in atom_types:
            atom_coords_of_a_given_type = get_coords_of_a_given_type_in_molecule(
                molecule, atom_type
            )

            # Transform the coordinates of each atom into eigen vector space such that the molecule is perpendicular
            # to z axis. The coordinates here are considered as 2x1 matrices
            atom_coords_of_a_given_type_in_evs_with_z_axed = (
                transform_coords_into_reduced_evs_for_planar_molecule(
                    atom_type, atom_coords_of_a_given_type, transf_mat_to_evs
                )
            )

            # The sym ops obtained by this routine are in the eigen vector space where molecule is perpendicular to
            # to z axis
            sym_ops = make_symops_from_same_atom_type_coords(
                atom_coords_of_a_given_type_in_evs_with_z_axed,
                atom_coords_with_atom_types_in_evs_with_z_nulled,
                True,
                tol,
            )

            for op in sym_ops:
                # convert back the symmetry operations into cartesian vector space
                new_op_in_old_space = np.linalg.inv(transf_mat_to_evs) @ op
                new_op_in_old_space = new_op_in_old_space @ transf_mat_to_evs
                if is_symop_unique(new_op_in_old_space, point_group, tol):
                    point_group.append(new_op_in_old_space)

        return point_group

    # If non planar
    # For each atom type
    for atom_type in atom_types:
        # Get coordinates of one atom type
        atom_coords_of_a_given_type = get_coords_of_a_given_type_in_molecule(
            molecule, atom_type
        )

        sym_ops = make_symops_from_same_atom_type_coords(
            atom_coords_of_a_given_type,
            atom_coords_along_with_atom_types,
            False,
            tol,
        )
        for sym_op in sym_ops:
            if is_symop_unique(sym_op, point_group, tol):
                point_group.append(sym_op)

    cluster_point_group = [
        xtal.SymOp(matrix=cart_op, translation=np.zeros(3), time_reversal=False)
        for cart_op in point_group
    ]

    return cluster_point_group
