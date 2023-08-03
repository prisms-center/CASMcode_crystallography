import numpy as np

import libcasm.xtal as xtal


def test_SymInfo_constructor():
    lattice = xtal.Lattice(np.eye(3))
    op = xtal.SymOp(np.eye(3), np.zeros((3, 1)), False)
    syminfo = xtal.SymInfo(op, lattice)
    assert syminfo.op_type() == "identity"
