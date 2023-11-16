from typing import Optional

import numpy as np

import libcasm.xtal as xtal
import libcasm.xtal.lattices as xtal_lattices


def cubic(
    a: Optional[float] = None,
    occ_dof: list[str] = ["A", "B"],
    local_dof: list[xtal.DoFSetBasis] = [],
    global_dof: list[xtal.DoFSetBasis] = [],
    occupants: dict[str, xtal.Occupant] = {},
    title: str = "prim",
) -> xtal.Prim:
    r"""Construct a simle cubic Prim

    Parameters
    ----------
    a: Optional[float] = None
        Specify the cubic lattice parameter :math:`a`.
    occ_dof : list[str] = ["A", "B"]
        Labels ('orientation names') of occupants allowed on the basis
        site. The values may either be (i) the name of an isotropic
        atom (i.e. "Mg") or vacancy ("Va"), or (ii) a key
        in the occupants dictionary (i.e. "H2O", or "H2_xx"). The names
        are case sensitive, and "Va" is reserved for vacancies.
    local_dof : list[xtal.DoFSetBasis]=[]
        Continuous DoF allowed on the basis site. No effect if empty.
    global_dof : list[xtal.DoFSetBasis]=[]
        Global continuous DoF allowed for the entire crystal.
    occupants : dict[str, xtal.Occupant]=[]
        :class:`~xtal.Occupant` allowed in the crystal. The keys are labels
        ('orientation names') used in the occ_dof parameter. This may
        include isotropic atoms, vacancies, atoms with fixed anisotropic
        properties, and molecular occupants. A seperate key and value is
        required for all species with distinct anisotropic properties
        (i.e. "H2_xy", "H2_xz", and "H2_yz" for distinct orientations,
        or "A.up", and "A.down" for distinct collinear magnetic spins,
        etc.).
    title : str="prim"
        A title for the prim. When the prim is used to construct a
        cluster expansion, this must consist of alphanumeric characters
        and underscores only. The first character may not be a number.

    Returns
    -------
    prim : xtal.Prim
        A simple cubic Prim
    """
    return xtal.Prim(
        lattice=xtal_lattices.cubic(a=a),
        coordinate_frac=np.array([[0.0, 0.0, 0.0]]).transpose(),
        occ_dof=[occ_dof],
        local_dof=[local_dof],
        global_dof=global_dof,
        occupants=occupants,
        title=title,
    )


def BCC(
    r: Optional[float] = None,
    a: Optional[float] = None,
    occ_dof: list[str] = ["A", "B"],
    local_dof: list[xtal.DoFSetBasis] = [],
    global_dof: list[xtal.DoFSetBasis] = [],
    occupants: dict[str, xtal.Occupant] = {},
    title: str = "prim",
) -> xtal.Prim:
    r"""Construct a BCC Prim

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct BCC primitive lattice
        with conventional lattice parameter :math:`a = 4r / \sqrt{3}`
    a: Optional[float] = None
        Specify the conventional BCC lattice parameter :math:`a`.
    occ_dof : list[str] = ["A", "B"]
        Labels ('orientation names') of occupants allowed on the basis
        site. The values may either be (i) the name of an isotropic
        atom (i.e. "Mg") or vacancy ("Va"), or (ii) a key
        in the occupants dictionary (i.e. "H2O", or "H2_xx"). The names
        are case sensitive, and "Va" is reserved for vacancies.
    local_dof : list[xtal.DoFSetBasis]=[]
        Continuous DoF allowed on the basis site. No effect if empty.
    global_dof : list[xtal.DoFSetBasis]=[]
        Global continuous DoF allowed for the entire crystal.
    occupants : dict[str, xtal.Occupant]=[]
        :class:`~xtal.Occupant` allowed in the crystal. The keys are labels
        ('orientation names') used in the occ_dof parameter. This may
        include isotropic atoms, vacancies, atoms with fixed anisotropic
        properties, and molecular occupants. A seperate key and value is
        required for all species with distinct anisotropic properties
        (i.e. "H2_xy", "H2_xz", and "H2_yz" for distinct orientations,
        or "A.up", and "A.down" for distinct collinear magnetic spins,
        etc.).
    title : str="prim"
        A title for the prim. When the prim is used to construct a
        cluster expansion, this must consist of alphanumeric characters
        and underscores only. The first character may not be a number.

    Returns
    -------
    prim : xtal.Prim
        A BCC Prim
    """
    return xtal.Prim(
        lattice=xtal_lattices.BCC(r=r, a=a),
        coordinate_frac=np.array([[0.0, 0.0, 0.0]]).transpose(),
        occ_dof=[occ_dof],
        local_dof=[local_dof],
        global_dof=global_dof,
        occupants=occupants,
        title=title,
    )


def FCC(
    r: Optional[float] = None,
    a: Optional[float] = None,
    occ_dof: list[str] = ["A", "B"],
    local_dof: list[xtal.DoFSetBasis] = [],
    global_dof: list[xtal.DoFSetBasis] = [],
    occupants: dict[str, xtal.Occupant] = {},
    title: str = "prim",
) -> xtal.Prim:
    r"""Construct a FCC Prim

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct FCC primitive lattice
        with conventional lattice parameter :math:`a = 4r / \sqrt{2}`
    a: Optional[float] = None
        Specify the conventional FCC lattice parameter :math:`a`.
    occ_dof : list[str] = ["A", "B"]
        Labels ('orientation names') of occupants allowed on the basis
        site. The values may either be (i) the name of an isotropic
        atom (i.e. "Mg") or vacancy ("Va"), or (ii) a key
        in the occupants dictionary (i.e. "H2O", or "H2_xx"). The names
        are case sensitive, and "Va" is reserved for vacancies.
    local_dof : list[xtal.DoFSetBasis]=[]
        Continuous DoF allowed on the basis site. No effect if empty.
    global_dof : list[xtal.DoFSetBasis]=[]
        Global continuous DoF allowed for the entire crystal.
    occupants : dict[str, xtal.Occupant]=[]
        :class:`~xtal.Occupant` allowed in the crystal. The keys are labels
        ('orientation names') used in the occ_dof parameter. This may
        include isotropic atoms, vacancies, atoms with fixed anisotropic
        properties, and molecular occupants. A seperate key and value is
        required for all species with distinct anisotropic properties
        (i.e. "H2_xy", "H2_xz", and "H2_yz" for distinct orientations,
        or "A.up", and "A.down" for distinct collinear magnetic spins,
        etc.).
    title : str="prim"
        A title for the prim. When the prim is used to construct a
        cluster expansion, this must consist of alphanumeric characters
        and underscores only. The first character may not be a number.

    Returns
    -------
    prim : xtal.Prim
        A FCC Prim
    """
    return xtal.Prim(
        lattice=xtal_lattices.FCC(r=r, a=a),
        coordinate_frac=np.array([[0.0, 0.0, 0.0]]).transpose(),
        occ_dof=[occ_dof],
        local_dof=[local_dof],
        global_dof=global_dof,
        occupants=occupants,
        title=title,
    )


def HCP(
    r: Optional[float] = None,
    a: Optional[float] = None,
    c: Optional[float] = None,
    occ_dof: list[str] = ["A", "B"],
    local_dof: list[xtal.DoFSetBasis] = [],
    global_dof: list[xtal.DoFSetBasis] = [],
    occupants: dict[str, xtal.Occupant] = {},
    title: str = "prim",
) -> xtal.Prim:
    r"""Construct a HCP Prim

    Parameters
    ----------
    r: Optional[float] = None
        Specify atomic radius and construct HCP primitive lattice
        with conventional lattice parameter :math:`a = 2r`.
    a: Optional[float] = None
        Specify the conventional HCP lattice parameter :math:`a`.
    c: Optional[float] = None
        Specify the conventional HCP lattice parameter :math:`c`.
        If not specified, the ideal value is used :math:`c = a \sqrt{8/3}`.
    occ_dof : list[str] = ["A", "B"]
        Labels ('orientation names') of occupants allowed on the basis
        sites. The values may either be (i) the name of an isotropic
        atom (i.e. "Mg") or vacancy ("Va"), or (ii) a key
        in the occupants dictionary (i.e. "H2O", or "H2_xx"). The names
        are case sensitive, and "Va" is reserved for vacancies.
    local_dof : list[xtal.DoFSetBasis]=[]
        Continuous DoF allowed on the basis sites. No effect if empty.
    global_dof : list[xtal.DoFSetBasis]=[]
        Global continuous DoF allowed for the entire crystal.
    occupants : dict[str, xtal.Occupant]=[]
        :class:`~xtal.Occupant` allowed in the crystal. The keys are labels
        ('orientation names') used in the occ_dof parameter. This may
        include isotropic atoms, vacancies, atoms with fixed anisotropic
        properties, and molecular occupants. A seperate key and value is
        required for all species with distinct anisotropic properties
        (i.e. "H2_xy", "H2_xz", and "H2_yz" for distinct orientations,
        or "A.up", and "A.down" for distinct collinear magnetic spins,
        etc.).
    title : str="prim"
        A title for the prim. When the prim is used to construct a
        cluster expansion, this must consist of alphanumeric characters
        and underscores only. The first character may not be a number.

    Returns
    -------
    prim : xtal.Prim
        A HCP Prim
    """
    return xtal.Prim(
        lattice=xtal_lattices.HCP(r=r, a=a, c=c),
        coordinate_frac=np.array(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.25],
                [2.0 / 3.0, 1.0 / 3.0, 0.75],
            ]
        ).transpose(),
        occ_dof=[occ_dof, occ_dof],
        local_dof=[local_dof, local_dof],
        global_dof=global_dof,
        occupants=occupants,
        title=title,
    )
