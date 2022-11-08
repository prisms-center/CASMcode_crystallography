#include "casm/crystallography/io/SimpleStructureIO.hh"

#include "autotools.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "gtest/gtest.h"
#include "testdir.hh"

using namespace CASM;

TEST(SimpleStructureFromJsonTest, Test1) {
  jsonParser json(test::data_file("crystallography", "structure_1.json"));
  xtal::SimpleStructure structure;
  from_json(structure, json);

  EXPECT_EQ(structure.atom_info.coords.cols(), 4);
  EXPECT_EQ(structure.atom_info.names,
            std::vector<std::string>({"A", "A", "B", "B"}));
  EXPECT_EQ(structure.mol_info.coords.cols(), 0);
  EXPECT_EQ(structure.mol_info.names.size(), 0);
  EXPECT_EQ(structure.properties.count("Hstrain"), 1);
  Eigen::MatrixXd Hstrain = structure.properties["Hstrain"];
  EXPECT_EQ(Hstrain.rows(), 6);
  EXPECT_EQ(Hstrain.cols(), 1);
}

TEST(SimpleStructureToJsonTest, Test1) {
  xtal::SimpleStructure structure;
  structure.lat_column_mat = Eigen::Matrix3d::Identity();
  structure.atom_info.resize(4);
  structure.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  jsonParser json;
  to_json(structure, json);
  EXPECT_EQ(json["atom_coords"].size(), 4);
  EXPECT_EQ(json["atom_type"].size(), 4);
  EXPECT_EQ(json.contains("mol_coords"), false);
  EXPECT_EQ(json.contains("mol_type"), false);
}
