from typing import Any, Union

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
