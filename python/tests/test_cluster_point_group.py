import os
import pytest
from typing import Tuple
import libcasm.xtal as xtal


def supply_molecules_and_expected_point_group_operations(
    shared_datadir: str, molecule_name: str
) -> Tuple[xtal.Occupant, str, float]:
    """Supplies a list of some general molecules on which
    you can test the point group code

    Parameters
    ----------
    session_shared_datadir : str
    molecule_name : str

    Returns
    -------
    Tuple[xtal.Occupant, int, tol]
        xtal.Occupant object, number of point group operations expected,
        tolerance of comparisions


    """
    molecules_dir = os.path.join(shared_datadir, "input_molecules")

    # For ammonia, you will have 6 point group operations
    if molecule_name == "ammonia":
        with open(os.path.join(molecules_dir, "ammonia.xyz"), "r") as f:
            xyz_string = f.read()
        return (
            xtal.Occupant.from_xyz_string(xyz_string),
            6,
            1e-3,
        )

    # For benzene, you will have 24 point group operations
    elif molecule_name == "benzene":
        with open(os.path.join(molecules_dir, "benzene.xyz"), "r") as f:
            xyz_string = f.read()
        return (
            xtal.Occupant.from_xyz_string(xyz_string),
            24,
            1e-3,
        )

    # For ethane, you will have 12 point group operations
    elif molecule_name == "ethane":
        with open(os.path.join(molecules_dir, "ethane.xyz"), "r") as f:
            xyz_string = f.read()
        return (
            xtal.Occupant.from_xyz_string(xyz_string),
            12,
            1e-2,
        )

    # For square, you will have 16 point group operations
    elif molecule_name == "square":
        with open(os.path.join(molecules_dir, "square.xyz"), "r") as f:
            xyz_string = f.read()
        return (
            xtal.Occupant.from_xyz_string(xyz_string),
            16,
            1e-5,
        )

    # For water, you will have 4 point group operations
    elif molecule_name == "water":
        with open(os.path.join(molecules_dir, "water.xyz"), "r") as f:
            xyz_string = f.read()
        return (
            xtal.Occupant.from_xyz_string(xyz_string),
            4,
            1e-5,
        )

    else:
        raise RuntimeError("Unknown molecule type")


@pytest.mark.parametrize(
    "molecule",
    [
        "ammonia",
        "benzene",
        "ethane",
        "square",
        "water",
    ],
)
def test_point_group_molecule(shared_datadir: str, molecule: str):
    """Tests the point group of molecule for the given code
    Currently tests against the hard coded number of point
    group operations a molecule should have.

    Parameters
    ----------
    pytest_root_dir : str
    molecule: str
    """
    (
        occupant,
        expected_number_of_pg_operations,
        tol,
    ) = supply_molecules_and_expected_point_group_operations(shared_datadir, molecule)

    point_group = xtal.cluster_point_group(occupant, tol)
    assert len(point_group) == expected_number_of_pg_operations
