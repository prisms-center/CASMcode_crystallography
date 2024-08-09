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


def test_DoFSetBasis_to_from_dict():
    disp_dof = xtal.DoFSetBasis(
        "disp", axis_names=["d_{1}"], basis=np.array([[1.0, 0.0, 0.0]]).transpose()
    )
    magspin_dof = xtal.DoFSetBasis("NCmagspin")
    site_dof = [disp_dof, magspin_dof]

    data = {}
    for dof in site_dof:
        dof.to_dict(data)

    assert disp_dof.dofname() in data
    assert "axis_names" in data[disp_dof.dofname()]
    assert disp_dof.axis_names() == data[disp_dof.dofname()]["axis_names"]
    assert np.allclose(
        np.array(data[disp_dof.dofname()]["basis"]),
        disp_dof.basis().transpose(),
    )

    assert magspin_dof.dofname() in data
    assert "axis_names" in data[magspin_dof.dofname()]
    assert magspin_dof.axis_names() == data[magspin_dof.dofname()]["axis_names"]
    assert np.allclose(
        np.array(data[magspin_dof.dofname()]["basis"]),
        magspin_dof.basis().transpose(),
    )


def test_DoFSetBasis_copy():
    import copy

    obj = xtal.DoFSetBasis(
        "disp", axis_names=["d_{1}"], basis=np.array([[1.0, 0.0, 0.0]]).transpose()
    )

    obj1 = obj.copy()
    assert isinstance(obj1, xtal.DoFSetBasis)
    assert obj1 is not obj

    obj2 = copy.copy(obj)
    assert isinstance(obj2, xtal.DoFSetBasis)
    assert obj2 is not obj

    obj3 = copy.deepcopy(obj)
    assert isinstance(obj3, xtal.DoFSetBasis)
    assert obj3 is not obj


def test_DoFSetBasis_repr():
    import io
    from contextlib import redirect_stdout

    disp_dof = xtal.DoFSetBasis(
        "disp", axis_names=["d_{1}"], basis=np.array([[1.0, 0.0, 0.0]]).transpose()
    )

    f = io.StringIO()
    with redirect_stdout(f):
        print(disp_dof)
    out = f.getvalue()
    assert "disp" in out
