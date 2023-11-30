import functools
import math
from collections import namedtuple
from typing import Any, Callable, Union

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
        arr: np.ndarray
            The array to be compared

        abs_tol: float = libcasm.casmglobal.TOL
            The absolute tolerance
        """
        if not isinstance(arr, np.ndarray):
            raise Exception("Error in ApproximateFloatArray: arr must be a np.ndarray")
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


def sort_structure_by_atom_info(
    structure: _xtal.Structure,
    key: Callable[[StructureAtomInfo], Any],
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only. Raises if
        ``len(structure.mol_type()) != 0``.
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
    """

    if len(structure.mol_type()) != 0:
        raise Exception(
            "Error: only atomic structures may be sorted using sort_by_atom_info"
        )
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

    atoms.sort(key=key, reverse=reverse)

    for i, atom in enumerate(atoms):
        atom_type[i] = atom[0]
        atom_coordinate_frac[:, i] = atom[1]
        for key in atom_properties:
            atom_properties[key][:, i] = atom[2][key]

    sorted_struc = _xtal.Structure(
        lattice=structure.lattice(),
        atom_type=atom_type,
        atom_coordinate_frac=atom_coordinate_frac,
        atom_properties=atom_properties,
        global_properties=structure.global_properties(),
    )

    return sorted_struc


def sort_structure_by_atom_type(
    structure: _xtal.Structure,
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure by atom type

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only. Raises if
        ``len(structure.mol_type()) != 0``.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by atom type.
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
        The structure to be sorted. Must be an atomic structure only. Raises if
        ``len(structure.mol_type()) != 0``.
    order: str = "cba"
        Sort order of fractional coordinate components. Default "cba" sorts by
        fractional coordinate along the "c" (third) lattice vector first, "b" (second)
        lattice vector second, and "a" (first) lattice vector third.
    abs_tol: float = libcasm.casmglobal.TOL
        Floating point tolerance for coordinate comparisons.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by fractional coordinates.
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
        The structure to be sorted. Must be an atomic structure only. Raises if
        ``len(structure.mol_type()) != 0``.
    order: str = "zyx"
        Sort order of Cartesian coordinate components. Default "zyx" sorts by
        "z" Cartesian coordinate first, "y" Cartesian coordinate second, and "x"
        Cartesian coordinate third.
    abs_tol: float = libcasm.casmglobal.TOL
        Floating point tolerance for coordinate comparisons.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by Cartesian coordinates.
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
