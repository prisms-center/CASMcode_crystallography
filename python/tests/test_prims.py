import libcasm.xtal.prims as prims
from libcasm.xtal import Prim


def test_construct_all_prims():
    assert isinstance(prims.BCC(r=1.0), Prim)
    assert isinstance(prims.BCC(a=1.0), Prim)

    assert isinstance(prims.FCC(r=1.0), Prim)
    assert isinstance(prims.FCC(a=1.0), Prim)

    assert isinstance(prims.HCP(r=1.0), Prim)
    assert isinstance(prims.HCP(r=1.0, c=2.0 * 1.8), Prim)
    assert isinstance(prims.HCP(a=1.0), Prim)
    assert isinstance(prims.HCP(a=1.0, c=1.8), Prim)
