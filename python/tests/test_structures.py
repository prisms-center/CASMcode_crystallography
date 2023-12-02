import numpy as np

import libcasm.xtal.structures as structures
from libcasm.xtal import Structure


def test_construct_all_structures():
    ## BCC ##
    assert isinstance(structures.BCC(r=1.0), Structure)
    assert isinstance(structures.BCC(a=1.0), Structure)

    structure = structures.BCC(a=1.0, conventional=True)
    assert isinstance(structure, Structure)
    assert np.allclose(structure.lattice().column_vector_matrix(), np.eye(3))
    assert structure.atom_type() == ["A", "A"]
    assert np.allclose(
        structure.atom_coordinate_frac(),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ]
        ).transpose(),
    )

    ## FCC ##
    assert isinstance(structures.FCC(r=1.0), Structure)
    assert isinstance(structures.FCC(a=1.0), Structure)

    structure = structures.FCC(a=1.0, conventional=True)
    assert isinstance(structure, Structure)
    assert np.allclose(structure.lattice().column_vector_matrix(), np.eye(3))
    assert structure.atom_type() == ["A", "A", "A", "A"]
    assert np.allclose(
        structure.atom_coordinate_frac(),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ).transpose(),
    )

    ## HCP ##
    assert isinstance(structures.HCP(r=1.0), Structure)
    assert isinstance(structures.HCP(r=1.0, c=2.0 * 1.8), Structure)
    assert isinstance(structures.HCP(a=1.0), Structure)
    assert isinstance(structures.HCP(a=1.0, c=1.8), Structure)
