#include "casm/crystallography/SymInfo.hh"

#include "TestStructures.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/crystallography/io/SymInfo_json_io.hh"
#include "casm/crystallography/io/SymInfo_stream_io.hh"
#include "gtest/gtest.h"

using namespace CASM;

TEST(SymInfoTest, Test1) {
  // TODO: improve test, currently just checks that it runs
  xtal::BasicStructure prim = test::FCC_binary_prim();

  auto fg = xtal::make_factor_group(prim);
  for (auto const &op : fg) {
    xtal::SymInfo syminfo{op, prim.lattice()};
    jsonParser json;
    to_json(syminfo, json);
    // std::cout << "syminfo json: " << std::endl << json << std::endl;
    // std::cout << "syminfo brief: " << to_brief_unicode(syminfo) << std::endl
    //           << std::endl;
  }
  EXPECT_TRUE(true);
}

TEST(SymInfoTest, Test2) {
  // TODO: improve test, currently just checks that it runs
  xtal::BasicStructure prim = test::SimpleCubic_ising_prim();

  auto fg = xtal::make_factor_group(prim);
  for (auto const &op : fg) {
    xtal::SymInfo syminfo{op, prim.lattice()};
    jsonParser json;
    to_json(syminfo, json);
    // std::cout << "syminfo: " << std::endl << json << std::endl;
    // std::cout << "syminfo brief: " << to_brief_unicode(syminfo) << std::endl
    //           << std::endl;
  }
  EXPECT_TRUE(true);
}

TEST(SymInfoTest, Test3) {
  // Check that the copy constructor sets the "home" lattice correctly for
  // all xtal::Coordinate in SymInfo
  xtal::BasicStructure prim = test::SimpleCubic_ising_prim();

  auto fg = xtal::make_factor_group(prim);

  xtal::SymInfo syminfo_first{fg.at(0), prim.lattice()};
  xtal::SymInfo syminfo_second(syminfo_first);
  EXPECT_EQ(&syminfo_second.axis.home(), &syminfo_second.lattice);
  EXPECT_EQ(&syminfo_second.screw_glide_shift.home(), &syminfo_second.lattice);
  EXPECT_EQ(&syminfo_second.location.home(), &syminfo_second.lattice);
}
