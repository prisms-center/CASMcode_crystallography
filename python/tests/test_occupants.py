import os
import numpy as np
from libcasm.xtal import Occupant


def test_occupant_from_xyz(shared_datadir):
    """Test making xtal.Occupant.from_xyz_string()
    method

    Parameters
    ----------
    shared_datadir : str

    Returns
    -------
    None

    """

    with open(os.path.join(shared_datadir, "input_molecules", "water.xyz"), "r") as f:
        water_xyz_string = f.read()

    water = Occupant.from_xyz_string(water_xyz_string)

    assert water.name() == "Water"

    expected_atom_names = ["O", "H", "H"]
    expected_atom_coords = [
        np.array([0.0, 0.0, 0.11779]),
        np.array([0.0, 0.75545, -0.47116]),
        np.array([0.0, -0.75545, -0.47116]),
    ]

    for atom, expected_atom_name, expected_atom_coord in zip(
        water.atoms(), expected_atom_names, expected_atom_coords
    ):

        assert atom.name() == expected_atom_name
        assert np.allclose(atom.coordinate(), expected_atom_coord)
