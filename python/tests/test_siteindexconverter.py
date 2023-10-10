import numpy as np
import pytest

import libcasm.xtal as xtal


def test_SiteIndexConverter_1():
    # test constructor
    f = xtal.SiteIndexConverter(
        transformation_matrix_to_super=np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        ),
        n_sublattice=2,
    )

    # test total_sites
    assert f.total_sites() == 16

    # test (b,i,j,k) -> linear_unitcell_index
    def bijk(b, i, j, k):
        return xtal.IntegralSiteCoordinate.from_list([b, i, j, k])

    assert f.linear_site_index(bijk(0, 0, 0, 0)) == 0
    assert f.linear_site_index(bijk(0, 1, 0, 0)) == 1
    assert f.linear_site_index(bijk(0, 0, 1, 0)) == 2
    assert f.linear_site_index(bijk(0, 1, 1, 0)) == 3
    assert f.linear_site_index(bijk(0, 0, 0, 1)) == 4
    assert f.linear_site_index(bijk(0, 1, 0, 1)) == 5
    assert f.linear_site_index(bijk(0, 0, 1, 1)) == 6
    assert f.linear_site_index(bijk(0, 1, 1, 1)) == 7
    assert f.linear_site_index(bijk(1, 0, 0, 0)) == 8
    assert f.linear_site_index(bijk(1, 1, 0, 0)) == 9
    assert f.linear_site_index(bijk(1, 0, 1, 0)) == 10
    assert f.linear_site_index(bijk(1, 1, 1, 0)) == 11
    assert f.linear_site_index(bijk(1, 0, 0, 1)) == 12
    assert f.linear_site_index(bijk(1, 1, 0, 1)) == 13
    assert f.linear_site_index(bijk(1, 0, 1, 1)) == 14
    assert f.linear_site_index(bijk(1, 1, 1, 1)) == 15

    assert f.integral_site_coordinate(0) == bijk(0, 0, 0, 0)
    assert f.integral_site_coordinate(1) == bijk(0, 1, 0, 0)
    assert f.integral_site_coordinate(2) == bijk(0, 0, 1, 0)
    assert f.integral_site_coordinate(3) == bijk(0, 1, 1, 0)
    assert f.integral_site_coordinate(4) == bijk(0, 0, 0, 1)
    assert f.integral_site_coordinate(5) == bijk(0, 1, 0, 1)
    assert f.integral_site_coordinate(6) == bijk(0, 0, 1, 1)
    assert f.integral_site_coordinate(7) == bijk(0, 1, 1, 1)
    assert f.integral_site_coordinate(8) == bijk(1, 0, 0, 0)
    assert f.integral_site_coordinate(9) == bijk(1, 1, 0, 0)
    assert f.integral_site_coordinate(10) == bijk(1, 0, 1, 0)
    assert f.integral_site_coordinate(11) == bijk(1, 1, 1, 0)
    assert f.integral_site_coordinate(12) == bijk(1, 0, 0, 1)
    assert f.integral_site_coordinate(13) == bijk(1, 1, 0, 1)
    assert f.integral_site_coordinate(14) == bijk(1, 0, 1, 1)
    assert f.integral_site_coordinate(15) == bijk(1, 1, 1, 1)

    # test explicit bring_within
    assert f.bring_within(bijk(0, 2, 2, 2)) == bijk(0, 0, 0, 0)

    # test default bring within
    assert f.linear_site_index(bijk(0, 2, 2, 2)) == 0

    # test never_bring_within
    with pytest.raises(RuntimeError):
        f.never_bring_within()
        assert f.linear_site_index(bijk(0, 2, 2, 2)) == 0

    # test always_bring_within
    f.never_bring_within()
    f.always_bring_within()
    assert f.linear_site_index(bijk(0, 2, 2, 2)) == 0


def test_SiteIndexConverter_2():
    # test constructor
    with pytest.raises(TypeError):
        xtal.SiteIndexConverter(
            transformation_matrix_to_super=np.array(
                [
                    [2.0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                ]
            ),
            n_sublattice=2,
        )
