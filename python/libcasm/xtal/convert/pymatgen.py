"""Convert between CASM and pymatgen dict formats (v2023.10.4)"""

# Note:
# - This package is intended to work on plain old Python data structures,
#   meaning it should work without needing to import libcasm, pymatgen,
#   ase, etc.
# - Use of standard Python modules and numpy is allowed

import copy
import re
from typing import Literal, Optional

import numpy as np


def make_pymatgen_lattice_dict(
    matrix: list[list[float]],
    pbc: tuple[bool] = (True, True, True),
) -> dict:
    """Create the pymatgen Lattice dict

    Parameters
    ----------
    matrix: list[list[float]]
        List of lattice vectors, (i.e. row-vector matrix of lattice vectors).
    pbc: tuple[bool] = (True, True, True)
        A tuple defining the periodic boundary conditions along the three
        axes of the lattice. Default is periodic in all directions. This
        is not stored in a CASM :class:`~_xtal.Lattice` and must be specified
        if it should be anything other than periodic along all three axes.

    Returns
    -------
    data: dict
        The pymatgen dict representation of a lattice, with format:

            matrix: list[list[float]]
                List of lattice vectors, (i.e. row-vector matrix of lattice vectors).

            pbc: tuple[bool] = (True, True, True)
                A tuple defining the periodic boundary conditions along the three
                axes of the lattice. Default is periodic in all directions.

    """
    return {
        "matrix": matrix,
        "pbc": pbc,
    }


# TODO:
# - def make_pymatgen_molecule_dict(casm_occupant: dict) -> dict
# - def make_casm_occupant_dict(pymatgen_molecule: dict) -> dict

# pymatgen.core.Site: dict
#   species: list[dict]
#       A list of species occupying the site, including occupation (float), and
#       optional "idealized" (integer) oxidation state and spin. Calculated (float)
#       charge and magnetic moments should be stored in Site.properties.
#
#       Ex:
#
#           [
#               {
#                   "element": "A",
#                   "occu": float,
#                   "oxidation_state": Optional[int],
#                   "spin": Optional[int],
#               },
#               ...
#           ]
#
#       Note that pymatgen expects "element" is an actual element in the periodic
#       table, or a dummy symbol which 'cannot have any part of first two letters that
#       will constitute an Element symbol. Otherwise, a composition may be parsed
#       wrongly. E.g., "X" is fine, but "Vac" is not because Vac contains V, a valid
#       Element.'.
#
#   xyz: list[float]
#       Cartesian coordinates
#   properties: Optional[dict]
#       Properties associated with the site. Options include: "magmom", ?
#   label: Optional[str]
#       Label for site

# pymatgen.core.IMolecule:
#   sites: list[Site]
#   charge: float = 0.0
#       Charge for the molecule.
#   spin_multiplicity: Optional[int] = None
#   properties: Optional[dict] = None
#       Properties associated with the molecule as a whole. Options include: ?

# def make_pymatgen_molecule_dict(
#     casm_occupant: dict,
# ) -> dict:
#     # charge =
#     # spin_multiplicity =
#     # sites =
#     # properties =
#     return {
#         "charge": charge,
#         "spin_multiplicity": spin_multiplicity,
#         "sites": sites,
#         "properties": properties,
#     }


def copy_properties(
    properties: dict,
    rename_as: dict[str, str] = {},
    include_all: bool = True,
) -> dict:
    """Copy a dictionary of properties, optionally renaming some

    Parameters
    ----------
    properties: dict
        The input properties

    rename_as: dict[str, str] = {}
        A lookup table where the keys are keys in the input `in_properties` that should
        be changed to the values in the output `out_properties`.

    include_all: bool = True
        If True, all properties in the input `in_properties` are included in the output
        `out_properties`. If False, only the properties found in `rename_as` are
        included in the output.

    Returns
    -------
    out_properties: dict
        A copy of `in_properties`, with the specified renaming of keys.

    """
    out_properties = {}
    for casm_key in properties:
        if include_all is False:
            if casm_key not in rename_as:
                continue
        new_key = rename_as.get(casm_key, casm_key)
        out_properties[new_key] = copy.deepcopy(properties[casm_key])
    return out_properties


def make_pymatgen_structure_dict(
    casm_structure: dict,
    charge: Optional[float] = None,
    pbc: tuple[bool] = (True, True, True),
    atom_type_to_pymatgen_species_list: dict = {},
    atom_type_to_pymatgen_label: dict = {},
    casm_to_pymatgen_atom_properties: dict = {},
    include_all_atom_properties: bool = True,
    casm_to_pymatgen_global_properties: dict = {},
    include_all_global_properties: bool = True,
) -> dict:
    """Convert a CASM :class:`~_xtal.Structure` dict to a pymatgen IStructure dict

    Parameters
    ----------
    structure: dict
        The :class:`~_xtal.Structure`, represented as a dict, to be represented as a
        pymatgen dict. Must be an atomic structure only.

    charge: Optional[float] = None
        Overall charge of the structure. If None, when pymatgen constructs an
        IStructure, default behavior is that the total charge is the sum of the
        oxidation states (weighted by occupation) on each site.

    pbc: tuple[bool] = (True, True, True)
        A tuple defining the periodic boundary conditions along the three
        axes of the lattice. Default is periodic in all directions. This
        is not stored in a CASM :class:`~_xtal.Lattice` and must be specified
        if it should be anything other than periodic along all three axes.

    atom_type_to_pymatgen_species_list: dict = {}
        A lookup table of CASM structure atom type (key) to pymatgen PeriodicSite
        species list (value) representing the occupancy of the site.

        The pymatgen species list format is:

        .. code-block:: Python

            [
                {
                    "element": str,
                    "occu": float,
                    "oxidation_state": Optional[int],
                    "spin": Optional[int],
                },
                ...
            ]

        where:

        - "element": str, the element name, or a "dummy" species name
        - "occu": float, the random site occupation
        - "oxidation_state": Optional[int], the oxidation state, expected to
          idealized to integer, e.g. -2, -1, 0, 1, 2, ...
        - "spin: Optional[int], the spin associated with the species

        By default, atoms in the input structure will be represented using
        ``[{ "element": atom_type, "occu": 1.0 }]``.

    atom_type_to_pymatgen_label: dict = {}
        A lookup table of CASM structure atom type (key) to pymatgen PeriodicSite label
        (value). If an atom_type in `casm_structure` is not found in the dict, then
        the atom_type is used for the label in the output `pymatgen_structure`.

    casm_to_pymatgen_atom_properties: dict = {}
        If a CASM structure atom property is found as a key in
        `casm_to_pymatgen_atom_properties`, it is renamed in the pymatgen
        PeriodicSite properties using the associated value.

    include_all_atom_properties: bool = True
        If True (default), all atom properties in `structure` are included in the
        result. If False , only `structure` atom properties found in
        `atom_properties_to_pymatgen_properties` are included in the result.

    casm_to_pymatgen_global_properties: dict = {}
        If a CASM structure global property is found as a key in
        `casm_to_pymatgen_global_properties`, it is renamed in the pymatgen
        structure properties using the associated value.

    include_all_global_properties: bool = True
        If True (default), all global properties in `structure` are included in the
        result. If False , only `structure` global properties found in
        `global_properties_to_pymatgen_properties` are included in the result.

    Returns
    -------
    data: dict
        The pymatgen IStructure `dict` representation, with format:

            sites: list[PeriodicSite]
                A list of PeriodicSite, with format:

                    species: list[dict]
                        A list of dict as described for the input parameter
                        `atom_type_to_pymatgen_species_list`.

                    abc: list[float]
                        Fractional coordinates of the site, relative to the lattice
                        vectors

                    properties: Optional[dict] = None
                        Properties associated with the site as a dict, e.g.
                        ``{"magmom": 5}``. Obtained from
                        `casm_structure.atom_properties`.

                    label: Optional[str] = None
                        Label for the site. Defaults to None.


            charge: Optional[float] = None
                Charge for the structure, expected to be equal to sum of oxidation
                states.

            lattice: dict
                Dict representation of a pymatgen Lattice, with format:

                    matrix: list[list[float]]
                        List of lattice vectors, (i.e. row-vector matrix of lattice
                        vectors).

                    pbc: tuple[bool] = (True, True, True)
                        A tuple defining the periodic boundary conditions along the
                        three axes of the lattice. Default is periodic in all
                        directions.

            properties: tuple[bool] = (True, True, True)
                Properties associated with the structure as a whole. Options include: ?

    Raises
    ------
    ValueError
        For non-atomic structure, if ``"mol_type" in structure``.
    """

    if "mol_type" in casm_structure:
        raise ValueError(
            "Error: only atomic structures may be converted using to_structure_dict"
        )

    ### Reading casm_structure

    # required keys: "lattice_vectors", "atom_type", "coordinate_mode", "atom_coords"
    lattice_column_vector_matrix = np.array(
        casm_structure["lattice_vectors"]
    ).transpose()
    atom_type = casm_structure["atom_type"]
    coordinate_mode = casm_structure["coordinate_mode"]

    if coordinate_mode in ["Fractional", "fractional", "FRAC", "Direct", "direct"]:
        atom_coordinate_frac = np.array(casm_structure["atom_coords"]).transpose()
    elif coordinate_mode in ["Cartesian", "cartesian", "CART"]:
        atom_coordinate_cart = np.array(casm_structure["atom_coords"]).transpose()
        atom_coordinate_frac = (
            np.linalg.pinv(lattice_column_vector_matrix) @ atom_coordinate_cart
        )
    else:
        raise Exception(f"Error: unrecognized coordinate_mode: {coordinate_mode}")

    atom_properties = {}
    if "atom_properties" in casm_structure:
        for key in casm_structure["atom_properties"]:
            atom_properties[key] = np.array(
                casm_structure["atom_properties"][key]["value"]
            ).transpose()

    global_properties = {}
    if "global_properties" in casm_structure:
        for key in casm_structure["global_properties"]:
            global_properties[key] = np.array(
                casm_structure["global_properties"][key]["value"]
            ).transpose()

    ### Convert to pymatgen dict

    # lattice
    lattice = {
        "matrix": lattice_column_vector_matrix.transpose().tolist(),
        "pbc": pbc,
    }

    # sites
    sites = []
    for i, _atom_type in enumerate(atom_type):
        # species list
        default_species_list = [{"element": _atom_type, "occu": 1.0}]
        species = atom_type_to_pymatgen_species_list.get(
            _atom_type, default_species_list
        )

        # abc coordinate
        abc = atom_coordinate_frac[:, i].tolist()

        # site properties
        _atom_properties = {}
        for key in atom_properties:
            _atom_properties[key] = atom_properties[:, i]
        properties = copy_properties(
            properties=_atom_properties,
            rename_as=casm_to_pymatgen_atom_properties,
            include_all=include_all_atom_properties,
        )

        # label
        label = atom_type_to_pymatgen_label.get(_atom_type, _atom_type)

        site = {
            "species": species,
            "abc": abc,
        }
        if label is not None:
            site["label"] = label
        if len(properties):
            site["properties"] = properties

        sites.append(site)

    # properties
    properties = copy_properties(
        properties=global_properties,
        rename_as=casm_to_pymatgen_global_properties,
        include_all=include_all_global_properties,
    )

    pymatgen_structure = {
        "lattice": lattice,
        "sites": sites,
    }
    if charge is not None:
        pymatgen_structure["charge"] = charge
    if len(properties):
        pymatgen_structure["properties"] = properties

    return pymatgen_structure


def make_casm_structure_dict(
    pymatgen_structure: dict,
    frac: bool = True,
    atom_type_from: Literal["element", "label", "species_list"] = "element",
    atom_type_to_pymatgen_species_list: dict = {},
    atom_type_to_pymatgen_label: dict = {},
    casm_to_pymatgen_atom_properties: dict = {},
    include_all_atom_properties: bool = True,
    casm_to_pymatgen_global_properties: dict = {},
    include_all_global_properties: bool = True,
) -> dict:
    """Convert a pymatgen IStructure dict to an atomic CASM :class:`~_xtal.Structure`
    dict

    Notes
    -----

    - An atomic CASM :class:`~_xtal.Structure` only allows one species at each basis
      site, whereas a pymatgen IStructure can allow a composition at each basis site.
    - The species at each site in the resulting CASM structure can be determined using
      one of several options chosen by a choice of the `atom_type_from` parameter.

    Parameters
    ----------
    pymatgen_structure: dict
        The pymatgen IStructure, represented as a dict, to be converted to a CASM
        :class:`~_xtal.Structure` dict representation. Must be an atomic structure only.

    frac: bool = True
        If True, coordinates in the result are expressed in fractional coordinates
        relative to the lattice vectors. Otherwise, Cartesian coordinates are used.

    atom_type_from: Literal["element", "label", "species_list"] = "element"
        Specifies which component of `pymatgen_structure` is used to determine the CASM
        structure `atom_type`. Options are:

        - "element": Use PeriodicSite species element name to determine the CASM
          `atom_type`.
        - "label": Use PeriodicSite "label" and `atom_type_to_pymatgen_label` to
          determine the CASM `atom_type`. With this option, a "label" must exist in
          the pymatgen PeriodicSite dict. By default, the label found is used for the
          atom type in the resulting CASM structure. If the label found is a value in
          the `atom_type_to_pymatgen_label` dict, then the associated key is used for
          the atom type in the resulting CASM structure.
        - "species_list": Use PeriodicSite species list to determine the CASM
          `atom_type` from the `atom_type_to_pymatgen_species_list` dict. The species
          list must compare equal to a value in the `atom_type_to_pymatgen_species_list`
          dict or the default value ``[{ "element": atom_type, "occu": 1.0 }]``,
          otherwise an exception will be raised.

    atom_type_to_pymatgen_species_list: dict = {}
        A lookup table of CASM structure atom type (key) to pymatgen PeriodicSite
        species list (value) representing the occupancy of the site.

        The pymatgen species list format is:

        .. code-block:: Python

            [
                {
                    "element": str,
                    "occu": float,
                    "oxidation_state": Optional[int],
                    "spin": Optional[int],
                },
                ...
            ]

        where:

        - "element": str, the element name, or a "dummy" species name
        - "occu": float, the random site occupation
        - "oxidation_state": Optional[int], the oxidation state, expected to
          idealized to integer, e.g. -2, -1, 0, 1, 2, ...
        - "spin: Optional[int], the spin associated with the species

        By default, atoms in the input structure will be represented using
        ``[{ "element": atom_type, "occu": 1.0 }]``.

    atom_type_to_pymatgen_label: dict = {}
        A lookup table of CASM structure atom type (key) to pymatgen PeriodicSite label
        (value).

    casm_to_pymatgen_atom_properties: dict = {}
        If a CASM structure atom property is found as a key in
        `casm_to_pymatgen_atom_properties`, it is renamed in the pymatgen
        PeriodicSite properties using the associated value.

    include_all_atom_properties: bool = True
        If True (default), all atom properties in `structure` are included in the
        result. If False , only `structure` atom properties found in
        `atom_properties_to_pymatgen_properties` are included in the result.

    casm_to_pymatgen_global_properties: dict = {}
        If a CASM structure global property is found as a key in
        `casm_to_pymatgen_global_properties`, it is renamed in the pymatgen
        structure properties using the associated value.

    include_all_global_properties: bool = True
        If True (default), all global properties in `structure` are included in the
        result. If False , only `structure` global properties found in
        `global_properties_to_pymatgen_properties` are included in the result.

    Returns
    -------
    data: dict
        The CASM Structure `dict` representation, with format described
        `here <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/SimpleStructure/>`_.
    """

    ### Reading pymatgen_structure

    # required keys: "lattice_vectors", "atom_type", "coordinate_mode", "atom_coords"
    lattice = pymatgen_structure["lattice"]["matrix"]
    # ? charge = pymatgen_structure.get("charge", None)
    sites = pymatgen_structure.get("sites", [])

    n_sites = len(sites)
    atom_type = []
    atom_coordinate_frac = np.zeros((3, n_sites))
    atom_properties = {}
    for i, site in enumerate(sites):
        # "species" / "abc" / "label" / "properties"

        # Determine the CASM atom_type associated with this site. The method
        # depends on the choice of the `atom_type_from`.
        if atom_type_from == "element":
            if len(site["species"]) != 1:
                raise Exception(
                    f"Error: multiple species on site {i}, which is not "
                    f'allowed with atom_type_from=="element"'
                )
            if "element" not in site["species"][0]:
                raise Exception(
                    f"Error: element not found for site {i}, "
                    f'which is not allowed with atom_type_from=="element"'
                )
            atom_type.append(site["species"][0]["element"])
        elif atom_type_from == "label":
            if "label" not in site:
                raise Exception(
                    f"Error: no label found on site {i}, which is not allowed "
                    f'with atom_type_from=="label"'
                )
            label = site.get("label")
            if label is None:
                raise Exception(
                    f"Error: label is null for site {i}, "
                    f'which is not allowed with atom_type_from=="label"'
                )
            _atom_type = None
            for key, value in atom_type_to_pymatgen_label.items():
                if value == label:
                    _atom_type = key
                    break
            if _atom_type is None:
                _atom_type = label
            atom_type.append(_atom_type)
        elif atom_type_from == "species_list":
            _atom_type = None
            for key, value in atom_type_to_pymatgen_species_list.items():
                if value == site["species"]:
                    _atom_type = key
                    break
            if _atom_type is None:
                if len(site["species"]) != 1:
                    raise Exception(
                        f'Error: no match found for the "species" list on site {i}, "'
                        f'which is not allowed with atom_type_from=="species_list"'
                    )
                element = site["species"][0].get("element", None)
                default_species_list = [{"element": element, "occu": 1.0}]
                if element is None or site["species"] != default_species_list:
                    raise Exception(
                        f'Error: no match found for the "species" list on site {i}, "'
                        f'which is not allowed with atom_type_from=="species_list"'
                    )
                _atom_type = element
            atom_type.append(_atom_type)
        else:
            raise Exception(f"Error: invalid atom_type_from=={atom_type_from}")

        atom_coordinate_frac[:, i] = np.array(site["abc"])

        if "properties" in site:
            for key in site["properties"]:
                _value = site["properties"][key]
                if isinstance(_value, [float, int]):
                    value = np.array([_value], dtype="float")
                elif isinstance(_value, list):
                    value = np.array([_value], dtype="float")
                else:
                    raise Exception(
                        f"Error: unsupported site properties: {str(_value)}"
                    )
                if len(value.shape) != 1:
                    raise Exception(
                        "Error: only scalar and vector site properties are supported"
                    )
                if key not in atom_properties:
                    atom_properties[key]["value"] = np.zeros((value.size, n_sites))
                atom_properties[key]["value"][:, i] = value

    global_properties = {}
    if "properties" in pymatgen_structure:
        for key in pymatgen_structure["properties"]:
            _value = pymatgen_structure["properties"][key]
            if isinstance(_value, [float, int]):
                value = np.array([_value], dtype="float")
            elif isinstance(_value, list):
                value = np.array([_value], dtype="float")
            else:
                raise Exception(
                    f"Error: unsupported global properties, {str(key)}:{str(_value)}"
                )
            if len(value.shape) != 1:
                raise Exception(
                    "Error: only scalar and vector global properties are supported"
                )
            global_properties[key]["value"] = value

    ### Convert to casm dict

    # ? "charge"

    if frac is True:
        coordinate_mode = "Fractional"
        atom_coords = atom_coordinate_frac
    else:
        coordinate_mode = "Cartesian"
        column_vector_matrix = np.array(lattice).transpose()
        atom_coords = column_vector_matrix @ atom_coordinate_frac

    atom_properties = copy_properties(
        properties=atom_properties,
        rename_as={
            value: key for key, value in casm_to_pymatgen_atom_properties.items()
        },
        include_all=include_all_atom_properties,
    )

    global_properties = copy_properties(
        properties=global_properties,
        rename_as={
            value: key for key, value in casm_to_pymatgen_global_properties.items()
        },
        include_all=include_all_global_properties,
    )

    casm_structure = {
        "lattice_vectors": lattice,
        "coordinate_mode": coordinate_mode,
        "atom_coords": atom_coords.transpose().tolist(),
        "atom_type": atom_type,
    }
    if len(atom_properties):
        casm_structure["atom_properties"] = atom_properties
    if len(global_properties):
        casm_structure["global_properties"] = global_properties

    return casm_structure


def make_casm_prim_dict(
    pymatgen_structure: dict,
    title: str = "Prim",
    description: Optional[str] = None,
    frac: bool = True,
    occupant_names_from: Literal["element", "species"] = "element",
    occupant_name_to_pymatgen_species: dict = {},
    occupant_sets: Optional[list[set[str]]] = None,
    use_site_labels: bool = False,
    casm_species: dict = {},
    site_dof: dict = {},
    global_dof: dict = {},
) -> dict:
    """Use a pymatgen IStructure dict to construct a CASM :class:`~_xtal.Prim` dict

    Notes
    -----

    - The allowed occupants on each site in the resulting CASM prim is determined from
      the elements on each site in the pymatgen structure.
    - This method does not attempt to use site properties or structure properties

    Parameters
    ----------
    pymatgen_structure: dict
        The pymatgen IStructure, represented as a dict, to be converted to a CASM
        :class:`~_xtal.Structure` dict representation. Must be an atomic structure only.

    title: str = "Prim"
        A title for the project. For use by CASM, should consist of alphanumeric
        characters and underscores only. The first character may not be a number.

    description: Optional[str] = None
        An extended description for the project. Included by convention in most example
        prim files, this attribute is not read by CASM.

    frac: bool = True
        If True, coordinates in the result are expressed in fractional coordinates
        relative to the lattice vectors. Otherwise, Cartesian coordinates are used.

    occupant_names_from: Literal["element", "species"] = "element"
        Specifies which component of `pymatgen_structure` is used to determine the CASM
        prim basis site occupant names. Options are:

        - "element": Use PeriodicSite species element name to determine the CASM
          occupant names.
        - "species": Use PeriodicSite species dicts to determine the CASM
          occupant names from the `occupant_name_to_pymatgen_species` dict. The species
          dict, excluding "occu", must compare equal to a value in the
          `occupant_name_to_pymatgen_species` dict or the default value
          ``[{ "element": occupant_name }]``, otherwise an exception will be raised.

        If a pymatgen site has mixed occupation, the corresponding CASM prim basis site
        will have multiple allowed occupants. Additionally, the `occupant_sets`
        parameter may be used to specify that the presence of one type of occupant
        specifies that a particular set of occupants should be allowed on the
        corresponding CASM prim basis sites.

    occupant_name_to_pymatgen_species: dict = {}
        A lookup table of CASM occupant name (key) to pymatgen PeriodicSite
        species (value) representing the occupancy of the site.

        The pymatgen species format is:

        .. code-block:: Python

            {
                "element": str,
                "oxidation_state": Optional[Union[int, float]],
                "spin": Optional[Union[int, float]],
            }

        where:

        - "element": str, the element name, or a "dummy" species name
        - "oxidation_state": Optional[Union[int, float]], the oxidation state, expected
          to be idealized to integer, e.g. -2, -1, 0, 1, 2, ...
        - "spin: Optional[Union[int, float]], the spin associated with the species

        Note that the "occu" part of the species representation on a PeriodicSite is
        ignored by this method.  The pymatgen documentation / implementation seem
        inconsistent whether these should be integer or float; for purposes of
        this method they just must compare exactly.

    occupant_sets: Optional[list[set[str]]] = None,
        Optional sets of occupants that should be allowed on the same sites.

        For example, if ``occupant_sets == [set(["A", "B"]), set(["C", "D"])]``, then
        any site where the pymatgen structure has either "A" or "B" occupation, the CASM
        prim basis site occupants will be expanded to include both "A" and "B", and any
        site where the pymatgen structure has either "C" or "D" occupation, the CASM
        prim basis site occupants will be expanded to include both "C" and "D".

    use_site_labels: bool = False
        If True, set CASM prim basis site labels based on the pymatgen structure site
        labels. If False, do not set prim basis site labels. CASM prim basis site labels
        are an integer, greater than or equal to zero, that if provided distinguishes
        otherwise identical sites.

    casm_species: dict = {},
        A dictionary used to define fixed properties of any species listed as an allowed
        occupant that is not a single isotropic atom. This parameter is filtered to
        only include occupants found in the input, then copied to the "species"
        attribute of the output prim dict.

    site_dof: dict = {}
        A dictionary specifying the types of continuous site degrees of freedom (DoF)
        allowed on every basis site. Note that CASM supports having different DoF on
        each basis site, but this method currently does not.

    global_dof: dict = {}
        A dictionary specifying the types of continuous global degrees of freedom (DoF)
        and their basis.

    Returns
    -------
    data: dict
        The CASM Prim `dict` representation, with format described
        `here <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/>`_.
        Site occupants are sorted in alphabetical order.
    """

    # validate title
    if not re.match("^[a-zA-Z]\w*$", title):
        raise Exception(f"Error: invalid title: {title}")

    ### Reading pymatgen_structure

    # required keys: "lattice_vectors", "atom_type", "coordinate_mode", "atom_coords"
    lattice = pymatgen_structure["lattice"]["matrix"]
    # ? charge = pymatgen_structure.get("charge", None)
    sites = pymatgen_structure.get("sites", [])

    n_sites = len(sites)
    occupants = []
    site_coordinate_frac = np.zeros((3, n_sites))
    label = []
    for i, site in enumerate(sites):
        # "species" / "abc" / "label" / "properties"

        # Determine the CASM occupant names associated with this site. The method
        # depends on the choice of the `occupant_names_from`.
        if occupant_names_from == "element":
            site_occupants = []
            for j, species in enumerate(site["species"]):
                element = species.get("element", None)
                if element is None:
                    raise Exception(
                        f'Error: no element for species {j} on site {i}, which is not "'
                        f'allowed with occupant_names_from=="element"'
                    )
                site_occupants.append(element)
            occupants.append(site_occupants)
        elif occupant_names_from == "species":
            site_occupants = []
            for j, species in enumerate(site["species"]):
                occupant_name = None
                _species = copy.deepcopy(species)
                if "occu" in _species:
                    del _species["occu"]
                # check occupant_name_to_pymatgen_species
                for key, value in occupant_name_to_pymatgen_species:
                    _value = copy.deepcopy(value)
                    if "occu" in _value:
                        del _value["occu"]
                    if value == _species:
                        occupant_name = key
                        break
                # check default, occupant_name == element (no oxidation_state, no spin)
                if occupant_name is None:
                    element = species.get("element", None)
                    default_species = [{"element": element}]
                    if _species == default_species:
                        occupant_name = element
                # if still not found, raise
                if occupant_name is None:
                    raise Exception(
                        f'Error: no match found for species {j} on site {i}, which is "'
                        f'not allowed with occupant_names_from=="species"'
                    )
                site_occupants.append(occupant_name)
            occupants.append(site_occupants)
        else:
            raise Exception(
                f"Error: invalid occupant_names_from=={occupant_names_from}"
            )

        site_coordinate_frac[:, i] = np.array(site["abc"])
        label.append(site.get("label", None))

    ### Convert to casm dict

    # ? "charge"

    # Get coordinate mode and coordinates
    if frac is True:
        coordinate_mode = "Fractional"
        site_coords = site_coordinate_frac
    else:
        coordinate_mode = "Cartesian"
        column_vector_matrix = np.array(lattice).transpose()
        site_coords = column_vector_matrix @ site_coordinate_frac

    # Expand site occupants based on occupant_sets
    if occupant_sets is not None:
        for i, site_occupants in enumerate(occupants):
            _site_occupants = set(site_occupants)

            for occ_name in site_occupants:
                for occ_set in occupant_sets:
                    if occ_name in occ_set:
                        _site_occupants.update(occ_set)

            occupants[i] = sorted(list(_site_occupants))

    # Get entries in casm_species that are actual occupants in the resulting prim
    filtered_casm_species = {}
    for site_occupants in occupants:
        for occupant_name in site_occupants:
            if occupant_name in casm_species:
                filtered_casm_species[occupant_name] = copy.deepcopy(
                    casm_species[occupant_name]
                )

    # Construct prim basis list
    basis = []
    distinct_site_labels = list(set(label))
    for i in range(n_sites):
        basis_site = {
            "coordinate": site_coords[:, i].tolist(),
            "occupants": occupants[i],
        }
        if use_site_labels:
            basis_site["label"] = distinct_site_labels.index(label[i])
        if len(site_dof):
            basis_site["dofs"] = (copy.deepcopy(site_dof),)
        basis.append(basis_site)

    # Construct prim dict
    prim = {
        "title": title,
        "lattice_vectors": lattice,
        "coordinate_mode": coordinate_mode,
        "basis": basis,
    }
    if description is not None:
        prim["description"] = description
    if len(global_dof):
        prim["dofs"] = copy.deepcopy(global_dof)
    if len(filtered_casm_species):
        prim["species"] = filtered_casm_species

    return prim
