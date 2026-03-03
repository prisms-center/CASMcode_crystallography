#include "casm/crystallography/BasicStructureTools.hh"

#include "TestStructures.hh"
#include "autotools.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "gtest/gtest.h"

using namespace CASM;

TEST(BasicStructureToolsTest, MakeAsymmetricUnitTest) {
  EXPECT_EQ(xtal::make_asymmetric_unit(test::no_basis_prim()),
            std::set<std::set<Index>>());
  EXPECT_EQ(xtal::make_asymmetric_unit(test::FCC_ternary_prim()),
            std::set<std::set<Index>>({{0}}));
  EXPECT_EQ(xtal::make_asymmetric_unit(test::ZrO_prim()),
            std::set<std::set<Index>>({{0, 1}, {2, 3}}));
}

TEST(BasicStructureToolsTest, MakeCrystalPointGroupTest) {
  EXPECT_EQ(xtal::make_crystal_point_group(
                xtal::make_factor_group(test::no_basis_prim()), TOL)
                .size(),
            48);
  EXPECT_EQ(xtal::make_crystal_point_group(
                xtal::make_factor_group(test::FCC_ternary_prim()), TOL)
                .size(),
            48);
  EXPECT_EQ(xtal::make_crystal_point_group(
                xtal::make_factor_group(test::ZrO_prim()), TOL)
                .size(),
            24);
}

TEST(BasicStructureToolsTest, MakeInternalTranslationsTest) {
  EXPECT_EQ(xtal::make_internal_translations(
                xtal::make_factor_group(test::no_basis_prim()), TOL)
                .size(),
            1);
  EXPECT_EQ(xtal::make_internal_translations(
                xtal::make_factor_group(test::FCC_ternary_prim()), TOL)
                .size(),
            1);
  EXPECT_EQ(xtal::make_internal_translations(
                xtal::make_factor_group(test::ZrO_prim()), TOL)
                .size(),
            1);
}
