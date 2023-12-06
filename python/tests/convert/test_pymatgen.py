import math

import numpy as np
import pytest

import libcasm.xtal.convert.pymatgen as convert

# Tests should work with or without libcasm.xtal
try:
    import libcasm.xtal as xtal

    import_libcasm_xtal = True
except ImportError:
    import_libcasm_xtal = False

# Tests should work with or without pymatgen
try:
    from pymatgen.core import IStructure

    import_pymatgen = True
except ImportError:
    import_pymatgen = False


def pretty_print(data):
    if import_libcasm_xtal:
        print(xtal.pretty_json(data))
    else:
        import json

        print(json.dumps(data, indent=2))


def test_pymatgen():
    """This just prints a message if pymatgen is not installed that we will be
    skipping some checks"""
    if not import_pymatgen:
        pytest.skip("Skipping checks that require pymatgen")


def test_pymatgen_compare():
    """Warning: order matters in comparison of IStructure

    For pymatgen v2023.11.12 the following asserts pass:
    """
    if import_pymatgen:
        lattice_vectors = [[5.692, 0.0, 0.0], [0.0, 5.692, 0.0], [0.0, 0.0, 5.692]]
        coords_frac = [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ]
        element_structure = IStructure(
            lattice=lattice_vectors,
            species=["Na"] * 4 + ["Cl"] * 4,
            coords=coords_frac,
            coords_are_cartesian=False,
        )
        species_structure = IStructure(
            lattice=lattice_vectors,
            species=["Na+"] * 4 + ["Cl-"] * 4,
            coords=coords_frac,
            coords_are_cartesian=False,
        )
        assert element_structure != species_structure
        assert species_structure == element_structure


def test_BCC_Fe_element():
    ### Make pymatgen structure from CASM structure
    casm_structure = {
        "atom_coords": [[0.0, 0.0, 0.0]],
        "atom_type": ["Fe"],
        "coordinate_mode": "Fractional",
        "lattice_vectors": [
            [-1.1547005383792517, 1.1547005383792517, 1.1547005383792517],
            [1.1547005383792517, -1.1547005383792517, 1.1547005383792517],
            [1.1547005383792517, 1.1547005383792517, -1.1547005383792517],
        ],
    }
    if import_libcasm_xtal:
        bcc_casm_structure = xtal.Structure.from_dict(casm_structure)
        assert isinstance(bcc_casm_structure, xtal.Structure)

    # Check pymatgen as_dict -> from_dict
    if import_pymatgen:
        bcc_pymatgen_structure = IStructure(
            lattice=casm_structure["lattice_vectors"],
            species=casm_structure["atom_type"],
            coords=casm_structure["atom_coords"],
            coords_are_cartesian=False,
        )
        # pretty_print(bcc_pymatgen_structure.as_dict())

    # Check libcasm.xtal.convert.to_pymatgen_structure_dict
    pymatgen_structure = convert.make_pymatgen_structure_dict(casm_structure)
    # pretty_print(pymatgen_structure)

    assert len(pymatgen_structure) == 2
    assert np.allclose(
        np.array(pymatgen_structure["lattice"]["matrix"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert pymatgen_structure["lattice"]["pbc"] == (True, True, True)
    assert "charge" not in pymatgen_structure
    assert "properties" not in pymatgen_structure
    assert len(pymatgen_structure["sites"]) == 1

    ## site 0
    site = pymatgen_structure["sites"][0]
    assert len(site) == 3
    assert np.allclose(site["abc"], np.array([0.0, 0.0, 0.0]))
    assert site["label"] == "Fe"
    assert "properties" not in site
    assert len(site["species"]) == 1

    # species 0
    species = site["species"][0]
    assert species["element"] == "Fe"
    assert math.isclose(species["occu"], 1.0)
    assert "oxidation_state" not in species

    if import_pymatgen:
        bcc_pymatgen_structure_in = IStructure.from_dict(pymatgen_structure)
        assert bcc_pymatgen_structure_in == bcc_pymatgen_structure
        assert bcc_pymatgen_structure == bcc_pymatgen_structure_in

    ### Make CASM structure from pymatgen structure
    casm_structure_2 = convert.make_casm_structure_dict(
        pymatgen_structure=pymatgen_structure,
        frac=True,
        atom_type_from="element",
    )
    # pretty_print(casm_structure_2)
    assert len(casm_structure_2) == 4
    assert np.allclose(
        np.array(casm_structure_2["lattice_vectors"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert casm_structure_2["atom_type"] == casm_structure["atom_type"]
    assert np.allclose(
        np.array(casm_structure_2["atom_coords"]),
        np.array(casm_structure["atom_coords"]),
    )
    assert casm_structure_2["coordinate_mode"] == "Fractional"
    assert "atom_properties" not in casm_structure_2
    assert "global_properties" not in casm_structure_2

    if import_libcasm_xtal:
        bcc_casm_structure_2 = xtal.Structure.from_dict(casm_structure)
        assert bcc_casm_structure.is_equivalent_to(bcc_casm_structure_2)

    ### Make CASM prim from pymatgen structure
    casm_prim = convert.make_casm_prim_dict(
        pymatgen_structure=pymatgen_structure,
        frac=True,
        occupant_names_from="element",
        occupant_sets=[set(["Fe", "Cr"])],
    )
    # pretty_print(casm_prim)
    assert len(casm_prim) == 4
    assert "title" in casm_prim
    assert "description" not in casm_prim
    assert np.allclose(
        np.array(casm_prim["lattice_vectors"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert casm_prim["coordinate_mode"] == "Fractional"
    assert "dofs" not in casm_prim

    assert len(casm_prim["basis"]) == 1
    basis_site = casm_prim["basis"][0]
    assert len(basis_site) == 2
    assert np.allclose(
        np.array(basis_site["coordinate"]), np.array(casm_structure["atom_coords"][0])
    )
    assert basis_site["occupants"] == ["Cr", "Fe"]

    if import_libcasm_xtal:
        bcc_casm_prim = xtal.Prim.from_dict(casm_prim)
        assert isinstance(bcc_casm_prim, xtal.Prim)


def test_NaCl_species_1():
    # This example constructs a CASM structure without oxidation states
    # and uses atom_type_to_pymatgen_species_list
    casm_structure = {
        "atom_coords": [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        "atom_type": ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        "coordinate_mode": "Fractional",
        "lattice_vectors": [[5.692, 0.0, 0.0], [0.0, 5.692, 0.0], [0.0, 0.0, 5.692]],
    }
    atom_type_to_pymatgen_species_list = {
        "Na": [{"element": "Na", "oxidation_state": 1, "occu": 1}],
        "Cl": [{"element": "Cl", "oxidation_state": -1, "occu": 1}],
    }
    pymatgen_species = ["Na+"] * 4 + ["Cl-"] * 4

    if import_libcasm_xtal:
        NaCl_casm_structure = xtal.Structure.from_dict(casm_structure)
        assert isinstance(NaCl_casm_structure, xtal.Structure)

    if import_pymatgen:
        NaCl_pymatgen_structure = IStructure(
            lattice=casm_structure["lattice_vectors"],
            species=pymatgen_species,
            coords=casm_structure["atom_coords"],
            coords_are_cartesian=False,
        )
        # pretty_print(NaCl_pymatgen_structure.as_dict())

    # Check libcasm.xtal.convert.to_pymatgen_structure_dict
    pymatgen_structure = convert.make_pymatgen_structure_dict(
        casm_structure,
        atom_type_to_pymatgen_species_list=atom_type_to_pymatgen_species_list,
    )
    # pretty_print(pymatgen_structure)

    assert len(pymatgen_structure) == 2
    assert np.allclose(
        np.array(pymatgen_structure["lattice"]["matrix"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert pymatgen_structure["lattice"]["pbc"] == (True, True, True)
    assert "charge" not in pymatgen_structure
    assert "properties" not in pymatgen_structure
    assert len(pymatgen_structure["sites"]) == 8

    for i, site in enumerate(pymatgen_structure["sites"]):
        # Na+ sites:
        if i < 4:
            assert len(site) == 3
            assert np.allclose(site["abc"], np.array(casm_structure["atom_coords"][i]))
            assert len(site["species"]) == 1
            assert site["label"] == "Na"

            # species 0
            species = site["species"][0]
            assert species["element"] == "Na"
            assert math.isclose(species["occu"], 1)
            assert math.isclose(species["oxidation_state"], 1)

        # Cl- sites:
        else:
            assert len(site) == 3
            assert np.allclose(site["abc"], np.array(casm_structure["atom_coords"][i]))
            assert len(site["species"]) == 1
            assert site["label"] == "Cl"

            # species 0
            species = site["species"][0]
            assert species["element"] == "Cl"
            assert math.isclose(species["occu"], 1)
            assert math.isclose(species["oxidation_state"], -1)

    if import_pymatgen:
        NaCl_pymatgen_structure_in = IStructure.from_dict(pymatgen_structure)
        # pretty_print(NaCl_pymatgen_structure_in.as_dict())
        assert NaCl_pymatgen_structure_in == NaCl_pymatgen_structure
        assert NaCl_pymatgen_structure == NaCl_pymatgen_structure_in

    ### Make CASM structure from pymatgen structure
    casm_structure_2 = convert.make_casm_structure_dict(
        pymatgen_structure=pymatgen_structure,
        frac=True,
        atom_type_from="species_list",
        atom_type_to_pymatgen_species_list=atom_type_to_pymatgen_species_list,
    )
    # pretty_print(casm_structure_2)
    assert len(casm_structure_2) == 4
    assert np.allclose(
        np.array(casm_structure_2["lattice_vectors"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert casm_structure_2["atom_type"] == casm_structure["atom_type"]
    assert np.allclose(
        np.array(casm_structure_2["atom_coords"]),
        np.array(casm_structure["atom_coords"]),
    )
    assert casm_structure_2["coordinate_mode"] == "Fractional"
    assert "atom_properties" not in casm_structure_2
    assert "global_properties" not in casm_structure_2

    if import_libcasm_xtal:
        NaCl_casm_structure_2 = xtal.Structure.from_dict(casm_structure)
        assert NaCl_casm_structure.is_equivalent_to(NaCl_casm_structure_2)

    ### Make CASM prim from pymatgen structure
    casm_prim = convert.make_casm_prim_dict(
        pymatgen_structure=pymatgen_structure,
        frac=True,
        occupant_names_from="element",
        occupant_sets=[set(["Na", "Va"]), set(["Cl", "Va"])],
    )
    # pretty_print(casm_prim)
    assert len(casm_prim) == 4
    assert "title" in casm_prim
    assert "description" not in casm_prim
    assert np.allclose(
        np.array(casm_prim["lattice_vectors"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert casm_prim["coordinate_mode"] == "Fractional"
    assert "dofs" not in casm_prim

    assert len(casm_prim["basis"]) == 8

    for i, basis_site in enumerate(casm_prim["basis"]):
        # Na+ sites:
        if i < 4:
            assert len(basis_site) == 2
            assert np.allclose(
                np.array(basis_site["coordinate"]),
                np.array(casm_structure["atom_coords"][i]),
            )
            assert basis_site["occupants"] == ["Na", "Va"]

        # Cl- sites:
        else:
            assert len(basis_site) == 2
            assert np.allclose(
                np.array(basis_site["coordinate"]),
                np.array(casm_structure["atom_coords"][i]),
            )
            assert basis_site["occupants"] == ["Cl", "Va"]

    if import_libcasm_xtal:
        NaCl_casm_prim = xtal.Prim.from_dict(casm_prim)
        assert isinstance(NaCl_casm_prim, xtal.Prim)


def test_NaCl_species_2():
    # This example uses atom_type_to_pymatgen_label to set labels and then also
    # tests using atom_types_from=="label" converting the pymatgen structure to a
    # casm structure
    casm_structure = {
        "atom_coords": [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        "atom_type": ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        "coordinate_mode": "Fractional",
        "lattice_vectors": [[5.692, 0.0, 0.0], [0.0, 5.692, 0.0], [0.0, 0.0, 5.692]],
    }
    atom_type_to_pymatgen_species_list = {
        "Na": [{"element": "Na", "oxidation_state": 1, "occu": 1}],
        "Cl": [{"element": "Cl", "oxidation_state": -1, "occu": 1}],
    }
    atom_type_to_pymatgen_label = {"Na": "Na+", "Cl": "Cl-"}
    pymatgen_species = ["Na+"] * 4 + ["Cl-"] * 4

    if import_libcasm_xtal:
        NaCl_casm_structure = xtal.Structure.from_dict(casm_structure)
        assert isinstance(NaCl_casm_structure, xtal.Structure)

    if import_pymatgen:
        NaCl_pymatgen_structure = IStructure(
            lattice=casm_structure["lattice_vectors"],
            species=pymatgen_species,
            coords=casm_structure["atom_coords"],
            coords_are_cartesian=False,
        )
        # pretty_print(NaCl_pymatgen_structure.as_dict())

    # Check libcasm.xtal.convert.to_pymatgen_structure_dict
    pymatgen_structure = convert.make_pymatgen_structure_dict(
        casm_structure,
        atom_type_to_pymatgen_species_list=atom_type_to_pymatgen_species_list,
        atom_type_to_pymatgen_label=atom_type_to_pymatgen_label,
    )
    # pretty_print(pymatgen_structure)

    assert len(pymatgen_structure) == 2
    assert np.allclose(
        np.array(pymatgen_structure["lattice"]["matrix"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert pymatgen_structure["lattice"]["pbc"] == (True, True, True)
    assert "charge" not in pymatgen_structure
    assert "properties" not in pymatgen_structure
    assert len(pymatgen_structure["sites"]) == 8

    for i, site in enumerate(pymatgen_structure["sites"]):
        # Na+ sites:
        if i < 4:
            assert len(site) == 3
            assert np.allclose(site["abc"], np.array(casm_structure["atom_coords"][i]))
            assert len(site["species"]) == 1
            assert site["label"] == "Na+"

            # species 0
            species = site["species"][0]
            assert species["element"] == "Na"
            assert math.isclose(species["occu"], 1)
            assert math.isclose(species["oxidation_state"], 1)

        # Cl- sites:
        else:
            assert len(site) == 3
            assert np.allclose(site["abc"], np.array(casm_structure["atom_coords"][i]))
            assert len(site["species"]) == 1
            assert site["label"] == "Cl-"

            # species 0
            species = site["species"][0]
            assert species["element"] == "Cl"
            assert math.isclose(species["occu"], 1)
            assert math.isclose(species["oxidation_state"], -1)

    if import_pymatgen:
        NaCl_pymatgen_structure_in = IStructure.from_dict(pymatgen_structure)
        # pretty_print(NaCl_pymatgen_structure_in.as_dict())
        assert NaCl_pymatgen_structure_in == NaCl_pymatgen_structure
        assert NaCl_pymatgen_structure == NaCl_pymatgen_structure_in

    ### Make CASM structure from pymatgen structure
    casm_structure_2 = convert.make_casm_structure_dict(
        pymatgen_structure=pymatgen_structure,
        frac=True,
        atom_type_from="label",
        atom_type_to_pymatgen_label=atom_type_to_pymatgen_label,
    )
    # pretty_print(casm_structure_2)
    assert len(casm_structure_2) == 4
    assert np.allclose(
        np.array(casm_structure_2["lattice_vectors"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert casm_structure_2["atom_type"] == casm_structure["atom_type"]
    assert np.allclose(
        np.array(casm_structure_2["atom_coords"]),
        np.array(casm_structure["atom_coords"]),
    )
    assert casm_structure_2["coordinate_mode"] == "Fractional"
    assert "atom_properties" not in casm_structure_2
    assert "global_properties" not in casm_structure_2

    if import_libcasm_xtal:
        NaCl_casm_structure_2 = xtal.Structure.from_dict(casm_structure)
        assert NaCl_casm_structure.is_equivalent_to(NaCl_casm_structure_2)

    ### Make CASM prim from pymatgen structure
    casm_prim = convert.make_casm_prim_dict(
        pymatgen_structure=pymatgen_structure,
        frac=True,
        occupant_names_from="element",
        occupant_sets=[set(["Na", "Va"]), set(["Cl", "Va"])],
    )
    # pretty_print(casm_prim)
    assert len(casm_prim) == 4
    assert "title" in casm_prim
    assert "description" not in casm_prim
    assert np.allclose(
        np.array(casm_prim["lattice_vectors"]),
        np.array(casm_structure["lattice_vectors"]),
    )
    assert casm_prim["coordinate_mode"] == "Fractional"
    assert "dofs" not in casm_prim

    assert len(casm_prim["basis"]) == 8

    for i, basis_site in enumerate(casm_prim["basis"]):
        # Na+ sites:
        if i < 4:
            assert len(basis_site) == 2
            assert np.allclose(
                np.array(basis_site["coordinate"]),
                np.array(casm_structure["atom_coords"][i]),
            )
            assert basis_site["occupants"] == ["Na", "Va"]

        # Cl- sites:
        else:
            assert len(basis_site) == 2
            assert np.allclose(
                np.array(basis_site["coordinate"]),
                np.array(casm_structure["atom_coords"][i]),
            )
            assert basis_site["occupants"] == ["Cl", "Va"]

    if import_libcasm_xtal:
        NaCl_casm_prim = xtal.Prim.from_dict(casm_prim)
        assert isinstance(NaCl_casm_prim, xtal.Prim)
