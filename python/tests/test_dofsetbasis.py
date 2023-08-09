import numpy as np
import pytest

import libcasm.xtal as xtal


def test_DoFSetBasis_constructor_disp():
    disp_dof = xtal.DoFSetBasis("disp")
    assert disp_dof.dofname() == "disp"
    assert len(disp_dof.axis_names()) == 3
    assert np.allclose(disp_dof.basis(), np.eye(3))


def test_DoFSetBasis_constructor_error():
    # len(axis_names) must equal basis.shape[1]
    with pytest.raises(RuntimeError):
        xtal.DoFSetBasis("disp", axis_names=["d_{1}"], basis=np.eye(3))


def test_DoFSetBasis_constructor_1d_disp():
    disp_dof = xtal.DoFSetBasis(
        "disp", axis_names=["d_{1}"], basis=np.array([[1.0, 0.0, 0.0]]).transpose()
    )
    assert disp_dof.dofname() == "disp"
    assert disp_dof.axis_names() == ["d_{1}"]
    assert np.allclose(disp_dof.basis(), np.array([[1.0, 0.0, 0.0]]).transpose())
