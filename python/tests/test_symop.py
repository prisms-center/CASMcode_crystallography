import pytest
import numpy as np
import libcasm.xtal as xtal


def test_SymOp_constructor():
    op = xtal.SymOp(np.eye(3), np.zeros((3, 1)), False)
    assert np.allclose(op.matrix(), np.eye(3))
    assert np.allclose(op.translation(), np.zeros((3, 1)))
    assert op.time_reversal() == False
