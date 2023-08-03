import libcasm.xtal.structures as structures
from libcasm.xtal import Structure


def test_construct_all_structures():
    assert isinstance(structures.BCC(r=1.0), Structure)
    assert isinstance(structures.BCC(a=1.0), Structure)

    assert isinstance(structures.FCC(r=1.0), Structure)
    assert isinstance(structures.FCC(a=1.0), Structure)

    assert isinstance(structures.HCP(r=1.0), Structure)
    assert isinstance(structures.HCP(r=1.0, c=2.0 * 1.8), Structure)
    assert isinstance(structures.HCP(a=1.0), Structure)
    assert isinstance(structures.HCP(a=1.0, c=1.8), Structure)
