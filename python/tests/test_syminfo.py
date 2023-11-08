import numpy as np

import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims


def test_SymInfo_constructor():
    lattice = xtal.Lattice(np.eye(3))
    op = xtal.SymOp(np.eye(3), np.zeros((3, 1)), False)
    syminfo = xtal.SymInfo(op, lattice)
    assert syminfo.op_type() == "identity"


def test_SymInfo_to_dict():
    xtal_prim = xtal_prims.BCC(r=1.0, occ_dof=["A"])
    factor_group = xtal.make_factor_group(xtal_prim)
    symgroup_info = []
    for op in factor_group:
        syminfo = xtal.SymInfo(op, xtal_prim.lattice())
        symgroup_info.append(
            {
                "info": syminfo.to_dict(),
                "op": op.to_dict(),
            }
        )
    assert len(symgroup_info) == 48
