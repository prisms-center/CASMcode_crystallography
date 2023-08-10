from typing import Optional

import numpy as np

import libcasm.xtal as xtal
import libcasm.xtal.lattices as xtal_lattices


def BCC(
    r: Optional[float] = None,
    a: Optional[float] = None,
    atom_type: str = "A",
    atom_properties: dict[str, np.ndarray] = {},
    global_properties: dict[str, np.ndarray] = {},
) -> xtal.Structure:
    r"""Construct a primitive BCC structure

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct BCC primitive lattice
        with conventional lattice parameter :math:`a = 4.0 * r / \sqrt{3.0}`
    a: Optional[float] = None
        Specify the conventional BCC lattice parameter :math:`a`.
    atom_type: str = "A"
        Atom type name
    atom_properties : Dict[str,  numpy.ndarray[numpy.float64[m, 1]]], default={}
        Continuous properties associated with the atom, if present. Keys must be the
        name of a CASM-supported property type. Values are arrays with dimensions
        matching the standard dimension of the property type.
    global_properties : Dict[str,  numpy.ndarray[numpy.float64[m, 1]]], default={}
        Continuous properties associated with entire crystal, if present. Keys must be
        the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.

    Returns
    -------
    structure : xtal.Structure
        A primitive BCC structure
    """
    return xtal.Structure(
        lattice=xtal_lattices.BCC(r=r, a=a),
        atom_coordinate_frac=np.array([0.0, 0.0, 0.0]),
        atom_type=[atom_type],
        atom_properties=atom_properties,
        global_properties=global_properties,
    )


def FCC(
    r: Optional[float] = None,
    a: Optional[float] = None,
    atom_type: str = "A",
    atom_properties: dict[str, np.ndarray] = {},
    global_properties: dict[str, np.ndarray] = {},
) -> xtal.Structure:
    r"""Construct a primitive FCC structure

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct FCC primitive lattice
        with conventional lattice parameter :math:`a = 4.0 * r / \sqrt{2.0}`
    a: Optional[float] = None
        Specify the conventional FCC lattice parameter :math:`a`.
    atom_type: str = "A"
        Atom type name
    atom_properties : Dict[str,  numpy.ndarray[numpy.float64[m, 1]]], default={}
        Continuous properties associated with the atom, if present. Keys must be the
        name of a CASM-supported property type. Values are arrays with dimensions
        matching the standard dimension of the property type.
    global_properties : Dict[str,  numpy.ndarray[numpy.float64[m, 1]]], default={}
        Continuous properties associated with entire crystal, if present. Keys must be
        the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.

    Returns
    -------
    structure : xtal.Structure
        A primitive FCC structure
    """
    return xtal.Structure(
        lattice=xtal_lattices.FCC(r=r, a=a),
        atom_coordinate_frac=np.array([0.0, 0.0, 0.0]),
        atom_type=[atom_type],
        atom_properties=atom_properties,
        global_properties=global_properties,
    )


def HCP(
    r: Optional[float] = None,
    a: Optional[float] = None,
    c: Optional[float] = None,
    atom_type: str = "A",
    atom_properties: dict[str, np.ndarray] = {},
    global_properties: dict[str, np.ndarray] = {},
) -> xtal.Structure:
    r"""Construct a primitive HCP structure

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct HCP primitive lattice
        with conventional lattice parameter :math:`a = 2.0 * r`.
    a: Optional[float] = None
        Specify the conventional HCP lattice parameter :math:`a`.
    c: Optional[float] = None
        Specify the conventional HCP lattice parameter :math:`c`.
        If not specified, the ideal value is used :math:`c = `a \sqrt{8./3.}`.
    atom_type: str = "A"
        Atom type name
    atom_properties : Dict[str,  numpy.ndarray[numpy.float64[m, 2]]], default={}
        Continuous properties associated with the atoms, if present. Keys must be the
        name of a CASM-supported property type. Values are arrays with dimensions
        matching the standard dimension of the property type.
    global_properties : Dict[str,  numpy.ndarray[numpy.float64[m, 1]]], default={}
        Continuous properties associated with entire crystal, if present. Keys must be
        the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.

    Returns
    -------
    structure : xtal.Structure
        A primitive HCP structure
    """
    return xtal.Structure(
        lattice=xtal_lattices.HCP(r=r, a=a, c=c),
        atom_coordinate_frac=np.array(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.25],
                [2.0 / 3.0, 1.0 / 3.0, 0.75],
            ]
        ).transpose(),
        atom_type=[atom_type, atom_type],
        atom_properties=atom_properties,
        global_properties=global_properties,
    )
