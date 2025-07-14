#include "casm/crystallography/version.hh"

#include <regex>

#include "casm/casm_io/container/stream_io.hh"
#include "casm/global/version.hh"
#include "gtest/gtest.h"

using namespace CASM;

namespace test {
std::regex semver_regex() {
  return std::regex(
      R"---((0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?)---");
}
}  // namespace test

TEST(VersionTest, TestGlobalVersion) {
  std::smatch v_match;
  std::regex_match(CASM::version(), v_match, test::semver_regex());

  EXPECT_EQ(v_match.size(), 6);
  std::vector<int> version_vec{std::stoi(v_match[1].str()),
                               std::stoi(v_match[2].str()),
                               std::stoi(v_match[3].str())};
  std::vector<int> min_version{2, 2, 0};
  EXPECT_TRUE(version_vec >= min_version)
      << "version: " << version_vec << " min_version: " << min_version
      << std::endl;
}

TEST(VersionTest, TestXtalVersion) {
  std::smatch v_match;
  std::regex_match(CASM::xtal::version(), v_match, test::semver_regex());

  // Use <major> "." <minor> "." <patch>
  // or <major> "." <minor> "." <patch> "-" <pre-release> using "alpha",
  // "beta.1", "beta.2", ...
  EXPECT_EQ(v_match.size(), 6);
  EXPECT_EQ(v_match[1].str(), "2");
  EXPECT_EQ(v_match[2].str(), "2");
  EXPECT_EQ(v_match[3].str(), "0");
  EXPECT_EQ(v_match[4].str(), "");

  EXPECT_EQ(CASM::xtal::version(), "2.2.0");
  EXPECT_EQ(CASM::xtal::version(), casm_crystallography_version());
}
