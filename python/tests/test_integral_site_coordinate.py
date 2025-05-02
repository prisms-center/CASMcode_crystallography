import numpy as np

import libcasm.xtal as xtal


def test_integral_site_coordinate_constructor(ZrO_prim):
    b = 0
    unitcell = np.array([0, 0, 0])
    integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
    assert isinstance(integral_site_coordinate, xtal.IntegralSiteCoordinate)
    assert b == integral_site_coordinate.sublattice()
    assert (unitcell == integral_site_coordinate.unitcell()).all()


def test_integral_site_coordinate_from_list(ZrO_prim):
    b = 0
    unitcell = np.array([1, 2, 3])
    integral_site_coordinate = xtal.IntegralSiteCoordinate.from_list([0, 1, 2, 3])
    assert b == integral_site_coordinate.sublattice()
    assert (unitcell == integral_site_coordinate.unitcell()).all()


def test_integral_site_coordinate_to_list(ZrO_prim):
    b = 0
    unitcell = np.array([1, 2, 3])
    integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
    assert integral_site_coordinate.to_list() == [0, 1, 2, 3]


def test_integral_site_coordinate_str(ZrO_prim):
    b = 0
    unitcell = np.array([1, 2, 3])
    integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
    assert str(integral_site_coordinate) == "[0, 1, 2, 3]"


def test_integral_site_coordinate_translate_add(ZrO_prim):
    b = 0
    unitcell = np.array([1, 2, 3])
    translation = np.array([0, 0, 1])

    integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
    translated = integral_site_coordinate + translation
    assert integral_site_coordinate.to_list() == [0, 1, 2, 3]
    assert translated.to_list() == [0, 1, 2, 4]


def test_integral_site_coordinate_translate_iadd(ZrO_prim):
    b = 0
    unitcell = np.array([1, 2, 3])
    translation = np.array([0, 0, 1])

    integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
    integral_site_coordinate += translation
    assert integral_site_coordinate.to_list() == [0, 1, 2, 4]


def test_integral_site_coordinate_translate_sub(ZrO_prim):
    b = 0
    unitcell = np.array([1, 2, 3])
    translation = np.array([0, 0, 1])

    integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
    translated = integral_site_coordinate - translation
    assert integral_site_coordinate.to_list() == [0, 1, 2, 3]
    assert translated.to_list() == [0, 1, 2, 2]


def test_integral_site_coordinate_translate_isub(ZrO_prim):
    b = 0
    unitcell = np.array([1, 2, 3])
    translation = np.array([0, 0, 1])

    integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
    integral_site_coordinate -= translation
    assert integral_site_coordinate.to_list() == [0, 1, 2, 2]


def test_integral_site_coordinate_compare(ZrO_prim):
    site1 = xtal.IntegralSiteCoordinate.from_list([0, 1, 2, 3])
    site2 = xtal.IntegralSiteCoordinate.from_list([0, 2, 2, 3])
    site3 = xtal.IntegralSiteCoordinate.from_list([1, 1, 2, 3])
    assert site1 < site2
    assert site3 < site2
    assert site1 < site3


def test_integral_site_coordinate_coordinate_cart(ZrO_prim):
    basis_coordinate_cart = ZrO_prim.coordinate_cart()
    N_basis = basis_coordinate_cart.shape[1]
    unitcell = np.array([0, 0, 0])
    for b in range(N_basis):
        integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
        assert np.allclose(
            integral_site_coordinate.coordinate_cart(ZrO_prim),
            basis_coordinate_cart[:, b],
        )


def test_integral_site_coordinate_coordinate_frac(ZrO_prim):
    basis_coordinate_frac = ZrO_prim.coordinate_frac()
    N_basis = basis_coordinate_frac.shape[1]
    unitcell = np.array([0, 0, 0])
    for b in range(N_basis):
        integral_site_coordinate = xtal.IntegralSiteCoordinate(b, unitcell)
        assert np.allclose(
            integral_site_coordinate.coordinate_frac(ZrO_prim),
            basis_coordinate_frac[:, b],
        )


def test_integral_site_coordinate_from_coordinate_cart(ZrO_prim):
    basis_coordinate_cart = ZrO_prim.coordinate_cart()
    N_basis = basis_coordinate_cart.shape[1]
    for b in range(N_basis):
        site = xtal.IntegralSiteCoordinate.from_coordinate_cart(
            basis_coordinate_cart[:, b], ZrO_prim
        )
        assert site.to_list() == [b, 0, 0, 0]


def test_integral_site_coordinate_from_coordinate_frac(ZrO_prim):
    basis_coordinate_frac = ZrO_prim.coordinate_frac()
    N_basis = basis_coordinate_frac.shape[1]
    for b in range(N_basis):
        site = xtal.IntegralSiteCoordinate.from_coordinate_frac(
            basis_coordinate_frac[:, b], ZrO_prim
        )
        assert site.to_list() == [b, 0, 0, 0]


def test_IntegralSiteCoordinateRep_1(ZrO_prim):
    from libcasm.counter import IntCounter

    prim = ZrO_prim
    factor_group = xtal.make_factor_group(prim)

    counter = IntCounter(
        initial=[0, 0, 0, 0],
        final=[len(prim.occ_dof()) - 1, 2, 2, 2],
        increment=[1, 1, 1, 1],
    )
    for x in counter:
        b = x[0]
        unitcell = np.array(x[1:])
        site = xtal.IntegralSiteCoordinate(b, unitcell)
        r_cart = site.coordinate_cart(prim)

        for op in factor_group:
            rep = xtal.IntegralSiteCoordinateRep(op, prim)
            site_after = rep * site
            assert np.allclose(
                site_after.coordinate_cart(prim),
                op * r_cart,
            )


def test_IntegralSiteCoordinateRep_copy(ZrO_prim):
    import copy

    prim = ZrO_prim
    factor_group = xtal.make_factor_group(prim)
    op = factor_group[1]
    obj = xtal.IntegralSiteCoordinateRep(op, prim)

    obj1 = obj.copy()
    assert isinstance(obj1, xtal.IntegralSiteCoordinateRep)
    assert obj1 is not obj

    obj2 = copy.copy(obj)
    assert isinstance(obj2, xtal.IntegralSiteCoordinateRep)
    assert obj2 is not obj

    obj3 = copy.deepcopy(obj)
    assert isinstance(obj3, xtal.IntegralSiteCoordinateRep)
    assert obj3 is not obj


def test_IntegralSiteCoordinateRep_repr(ZrO_prim):
    prim = ZrO_prim
    factor_group = xtal.make_factor_group(prim)

    import io
    from contextlib import redirect_stdout

    op = factor_group[1]
    rep = xtal.IntegralSiteCoordinateRep(op, prim)

    f = io.StringIO()
    with redirect_stdout(f):
        print(rep)
    out = f.getvalue()
    assert "matrix_frac" in out
