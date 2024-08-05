import functools
import math
from collections import namedtuple
from typing import Any, Callable, Optional, Union

import numpy as np

import libcasm.casmglobal
import libcasm.xtal._xtal as _xtal


def make_primitive(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> Any:
    """Make the primitive cell of a Prim or atomic Structure

    Notes
    -----
    Currently, for Structure this method only considers atom coordinates and types.
    Molecular coordinates and types are not considered. Properties are not considered.
    The default CASM tolerance is used for comparisons. To consider molecules
    or properties, or to use a different tolerance, use a Prim.

    Parameters
    ----------
    obj: Union[ _xtal.Prim, _xtal.Structure]
        A Prim or an atomic Structure, which determines whether
        :func:`~libcasm.xtal.make_primitive_prim`, or
        :func:`~libcasm.xtal.make_primitive_structure` is called.

    Returns
    -------
    canonical_obj : Union[_xtal.Prim, _xtal.Structure]
        The primitive equivalent Prim or atomic Structure.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_primitive_prim(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_primitive_structure(obj)
    else:
        raise TypeError(f"TypeError in make_primitive: received {type(obj).__name__}")


def make_canonical(
    obj: Union[_xtal.Lattice, _xtal.Prim, _xtal.Structure],
) -> Any:
    """Make an equivalent Lattice, Prim, or Structure with the canonical form
    of the lattice

    Parameters
    ----------
    obj: Union[_xtal.Lattice, _xtal.Prim, _xtal.Structure]
        A Lattice, Prim, or Structure, which determines whether
        :func:`~libcasm.xtal.make_canonical_lattice`, or
        :func:`~libcasm.xtal.make_canonical_prim`,
        :func:`~libcasm.xtal.make_canonical_structure` is called.

    Returns
    -------
    canonical_obj : Union[_xtal.Lattice, _xtal.Prim, _xtal.Structure]
        The equivalent Lattice, Prim, or Structure with canonical form of the lattice.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_canonical_prim(obj)
    elif isinstance(obj, _xtal.Lattice):
        return _xtal.make_canonical_lattice(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_canonical_structure(obj)
    else:
        raise TypeError(f"TypeError in make_canonical: received {type(obj).__name__}")


def make_crystal_point_group(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> list[_xtal.SymOp]:
    """Make the crystal point group of a Prim or Structure

    Parameters
    ----------
    obj: Union[_xtal.Prim, _xtal.Structure]
        A Prim or Structure, which determines whether
        :func:`~libcasm.xtal.make_prim_crystal_point_group` or
        :func:`~libcasm.xtal.make_structure_crystal_point_group` is called.

    Returns
    -------
    crystal_point_group : list[:class:`~libcasm.xtal.SymOp`]
        The crystal point group is the group constructed from the factor
        group operations with translation vector set to zero.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_prim_crystal_point_group(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_structure_crystal_point_group(obj)
    else:
        raise TypeError(
            f"TypeError in make_crystal_point_group: received {type(obj).__name__}"
        )


def make_factor_group(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> list[_xtal.SymOp]:
    """Make the factor group of a Prim or Structure

    Notes
    -----
    For :class:`~libcasm.xtal.Structure`, this method only considers atom coordinates
    and types. Molecular coordinates and types are not considered. Properties are not
    considered. The default CASM tolerance is used for comparisons. To consider
    molecules or properties, or to use a different tolerance, use a
    :class:`~libcasm.xtal.Prim` with :class:`~libcasm.xtal.Occupant` that have
    properties.

    Parameters
    ----------
    obj: Union[_xtal.Prim, _xtal.Structure]
        A Prim or Structure, which determines whether
        :func:`~libcasm.xtal.make_prim_factor_group` or
        :func:`~libcasm.xtal.make_structure_factor_group` is called.

    Returns
    -------
    factor_group : list[:class:`~libcasm.xtal.SymOp`]
        The set of symmery operations, with translation lying within the
        primitive unit cell, that leave the lattice vectors, global DoF
        (for :class:`~libcasm.xtal.Prim`), and basis site coordinates and local DoF
        (for :class:`~libcasm.xtal.Prim`) or atom coordinates and atom types
        (for :class:`~libcasm.xtal.Structure`) invariant.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_prim_factor_group(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_structure_factor_group(obj)
    else:
        raise TypeError(
            f"TypeError in make_factor_group: received {type(obj).__name__}"
        )


def make_within(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> Any:
    """Returns an equivalent Prim or Structure with all site coordinates within the \
    unit cell

    Parameters
    ----------
    obj: Union[_xtal.Prim, _xtal.Structure]
        A Prim or Structure, which determines whether
        :func:`~libcasm.xtal.make_prim_within` or
        :func:`~libcasm.xtal.make_structure_within` is called.

    Returns
    -------
    obj_within : Any
        An equivalent Prim or Structure with all site coordinates within the \
        unit cell.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_prim_within(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_structure_within(obj)
    else:
        raise TypeError(f"TypeError in make_within: received {type(obj).__name__}")


@functools.total_ordering
class ApproximateFloatArray:
    def __init__(
        self,
        arr: np.ndarray,
        abs_tol: float = libcasm.casmglobal.TOL,
    ):
        """Store an array that will be compared lexicographically up to a given
        absolute tolerance using math.isclose

        Parameters
        ----------
        arr: numpy.ndarray
            The array to be compared

        abs_tol: float = :data:`~libcasm.casmglobal.TOL`
            The absolute tolerance
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                "Error in ApproximateFloatArray: arr must be a numpy.ndarray"
            )
        self.arr = arr
        self.abs_tol = abs_tol

    def __eq__(self, other):
        if len(self.arr) != len(other.arr):
            return False
        for i in range(len(self.arr)):
            if not math.isclose(self.arr[i], other.arr[i], abs_tol=self.abs_tol):
                return False
        return True

    def __lt__(self, other):
        if len(self.arr) != len(other.arr):
            return len(self.arr) < len(other.arr)
        for i in range(len(self.arr)):
            if not math.isclose(self.arr[i], other.arr[i], abs_tol=self.abs_tol):
                return self.arr[i] < other.arr[i]
        return False


StructureAtomInfo = namedtuple(
    "StructureAtomInfo",
    ["atom_type", "atom_coordinate_frac", "atom_coordinate_cart", "atom_properties"],
)
""" A namedtuple, used to hold atom info when sorting, filtering, etc. atoms in a 
:class:`~_xtal.Structure`.

.. rubric:: Constructor

Parameters
----------
atom_type: str
    The atom type, from :func:`~_xtal.Structure.atom_type`.
atom_coordinate_frac: numpy.ndarray[numpy.float64[3]]
    The fractional coordinate of the atom, from 
    :func:`~_xtal.Structure.atom_type.atom_coordinate_frac`.
atom_coordinate_cart: numpy.ndarray[numpy.float64[3]]
    The Cartesian coordinate of the atom, from 
    :func:`~_xtal.Structure.atom_type.atom_coordinate_cart`.
atom_properties: dict[str, numpy.ndarray[numpy.float64[m]]]
    The continuous properties associated with the atoms, if present, from 
    :func:`~_xtal.Structure.atom_type.atom_coordinate_frac`. All atoms
    must have the same properties with values of the same dimension.
"""


def make_structure_atom_info(
    structure: _xtal.Structure,
) -> list[StructureAtomInfo]:
    """Create a list of StructureAtomInfo from a Structure

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted, filtered, etc. by atom info.

    Returns
    -------
    structure_atom_info: list[StructureAtomInfo]
        A list of StructureAtomInfo.

    """

    atom_type = structure.atom_type()
    atom_coordinate_frac = structure.atom_coordinate_frac()
    atom_coordinate_cart = structure.atom_coordinate_cart()
    atom_properties = structure.atom_properties()

    atoms = []
    import copy

    for i in range(len(atom_type)):
        atoms.append(
            StructureAtomInfo(
                copy.copy(atom_type[i]),
                atom_coordinate_frac[:, i].copy(),
                atom_coordinate_cart[:, i].copy(),
                {key: atom_properties[key][:, i].copy() for key in atom_properties},
            )
        )

    return atoms


def make_structure_from_atom_info(
    lattice: _xtal.Lattice,
    atoms: list[StructureAtomInfo],
    global_properties: dict[str, np.ndarray[np.float64]] = {},
) -> _xtal.Structure:
    """Create a Structure from a list of StructureAtomInfo

    Parameters
    ----------
    lattice: _xtal.Lattice]
        The lattice for the resulting structure.
    atoms: list[StructureAtomInfo]
        A list of StructureAtomInfo. The Cartesian coordinates are used when setting
        atom coordinates in the resulting structure.
    global_properties: dict[str, numpy.ndarray[numpy.float64[m, n]]] = {}
        Continuous properties associated with entire crystal, if present. Keys must be
        the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.

    Returns
    -------
    structure: _xtal.Structure
        The resulting structure
    """

    n_atoms = len(atoms)

    atom_type = [atom.atom_type for atom in atoms]
    atom_coordinate_cart = np.zeros((3, n_atoms))
    atom_properties = {}

    for i, atom in enumerate(atoms):
        if i == 0:
            for key, value in atom.atom_properties.items():
                dim = value.shape[0]
                atom_properties[key] = np.zeros((dim, n_atoms))

        atom_coordinate_cart[:, i] = atom.atom_coordinate_cart
        for key, value in atom.atom_properties.items():
            atom_properties[key][:, i] = atom.atom_properties[key]

    atom_coordinate_frac = _xtal.cartesian_to_fractional(
        lattice=lattice,
        coordinate_cart=atom_coordinate_cart,
    )

    return _xtal.Structure(
        lattice=lattice,
        atom_type=atom_type,
        atom_coordinate_frac=atom_coordinate_frac,
        atom_properties=atom_properties,
        global_properties=global_properties,
    )


def sort_structure_by_atom_info(
    structure: _xtal.Structure,
    key: Callable[[StructureAtomInfo], Any],
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    key: Callable[[StructureAtomInfo], Any]
        The function used to return a value which is sorted. This is passed to the
        `key` parameter of `list.sort()` to sort a `list[StructureAtomInfo]`.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted as specified.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """

    if len(structure.mol_type()) != 0:
        raise ValueError(
            "Error: only atomic structures may be sorted using sort_by_atom_info"
        )

    atoms = make_structure_atom_info(structure)
    atoms.sort(key=key, reverse=reverse)

    return make_structure_from_atom_info(
        lattice=structure.lattice(),
        atoms=atoms,
        global_properties=structure.global_properties(),
    )


def sort_structure_by_atom_type(
    structure: _xtal.Structure,
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure by atom type

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by atom type.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """
    return sort_structure_by_atom_info(
        structure,
        key=lambda atom_info: atom_info.atom_type,
        reverse=reverse,
    )


def sort_structure_by_atom_coordinate_frac(
    structure: _xtal.Structure,
    order: str = "cba",
    abs_tol: float = libcasm.casmglobal.TOL,
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure by fractional coordinates

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    order: str = "cba"
        Sort order of fractional coordinate components. Default "cba" sorts by
        fractional coordinate along the "c" (third) lattice vector first, "b" (second)
        lattice vector second, and "a" (first) lattice vector third.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        Floating point tolerance for coordinate comparisons.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by fractional coordinates.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """

    def compare_f(atom_info):
        values = []
        for i in range(len(order)):
            if order[i] == "a":
                values.append(atom_info.atom_coordinate_frac[0])
            elif order[i] == "b":
                values.append(atom_info.atom_coordinate_frac[1])
            elif order[i] == "c":
                values.append(atom_info.atom_coordinate_frac[2])

        return ApproximateFloatArray(
            arr=np.array(values),
            abs_tol=abs_tol,
        )

    return sort_structure_by_atom_info(
        structure,
        key=compare_f,
        reverse=reverse,
    )


def sort_structure_by_atom_coordinate_cart(
    structure: _xtal.Structure,
    order: str = "zyx",
    abs_tol: float = libcasm.casmglobal.TOL,
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure by Cartesian coordinates

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    order: str = "zyx"
        Sort order of Cartesian coordinate components. Default "zyx" sorts by
        "z" Cartesian coordinate first, "y" Cartesian coordinate second, and "x"
        Cartesian coordinate third.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        Floating point tolerance for coordinate comparisons.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by Cartesian coordinates.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """

    def compare_f(atom_info):
        values = []
        for i in range(len(order)):
            if order[i] == "x":
                values.append(atom_info.atom_coordinate_frac[0])
            elif order[i] == "y":
                values.append(atom_info.atom_coordinate_frac[1])
            elif order[i] == "z":
                values.append(atom_info.atom_coordinate_frac[2])

        return ApproximateFloatArray(
            arr=np.array(values),
            abs_tol=abs_tol,
        )

    return sort_structure_by_atom_info(
        structure,
        key=compare_f,
        reverse=reverse,
    )


def substitute_structure_species(
    structure: _xtal.Structure,
    substitutions: dict[str, str],
) -> _xtal.Structure:
    """Create a copy of a structure with renamed atomic and molecular species

    Parameters
    ----------
    structure: _xtal.Structure
        The initial structure
    substitutions: dict[str, str]
        The substitutions to make, using the convention key->value. For example, using
        ``substitutions = { "B": "C"}`` results in all `atom_type` and `mol_type`
        equal to "B" in the input structure being changed to "C" in the output
        structure.

    Returns
    -------
    structure_with_substitutions: _xtal.Structure
        A copy of `structure`, with substitutions of `atom_type` and `mol_type`.
    """
    return _xtal.Structure(
        lattice=structure.lattice(),
        atom_coordinate_frac=structure.atom_coordinate_frac(),
        atom_type=[substitutions.get(x, x) for x in structure.atom_type()],
        atom_properties=structure.atom_properties(),
        mol_coordinate_frac=structure.mol_coordinate_frac(),
        mol_type=[substitutions.get(x, x) for x in structure.mol_type()],
        mol_properties=structure.mol_properties(),
        global_properties=structure.global_properties(),
    )


def filter_structure_by_atom_info(
    structure: _xtal.Structure,
    filter: Callable[[StructureAtomInfo], Any],
) -> _xtal.Structure:
    """Return a copy of a structure with atoms passing a filter function

    .. rubric:: Example usage

    .. code-block:: Python

        # Remove all atoms with z coordinate >= 2.0
        structure_without_Al = filter_structure_by_atom_info(
            input_structure,
            lambda atom_info: atom_info.atom_coordinate_cart[2] < 2.0,
        )

    Parameters
    ----------
    structure: _xtal.Structure
        The initial structure
    filter: Callable[[StructureAtomInfo], Any]
        A function of `StructureAtomInfo` which returns True for atoms that should be
        kept and False for atoms that should be removed.

    Returns
    -------
    structure_after_removals: _xtal.Structure
        A copy of `structure` excluding atoms for which the `filter` function returns
        False.
    """
    return make_structure_from_atom_info(
        lattice=structure.lattice(),
        atoms=[x for x in make_structure_atom_info(structure) if filter(x)],
        global_properties=structure.global_properties(),
    )


def combine_structures(
    structures: list[_xtal.Structure],
    lattice: Optional[_xtal.Lattice] = None,
    global_properties: dict[str, np.ndarray[np.float64]] = {},
) -> _xtal.Structure:
    """Return a new structure which combines the atomic species of all input structures

    Parameters
    ----------
    structures: list[_xtal.Structure]
        The structures to be combined.
    lattice: Optional[_xtal.Lattice] = None
        If not None, the lattice of the resulting structure. If None, the resulting
        structure will use the lattice of the first structure.
    global_properties: dict[str, numpy.ndarray[numpy.float64[m, n]]] = {}
        Continuous properties associated with entire crystal, if present. Keys must be
        the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.

    Returns
    -------
    combined_structure: _xtal.Structure
        The resulting structure, which contains the species of all input structures
        with Cartesian coordinates fixed to the same values as in the input structures.
        All atom or molecule properties remain the same as in the input structures.
    """
    atoms = []
    for structure in structures:
        if lattice is None:
            lattice = structure.lattice()
        atoms += make_structure_atom_info(structure)
    return make_structure_from_atom_info(
        lattice=lattice,
        atoms=atoms,
        global_properties=global_properties,
    )
