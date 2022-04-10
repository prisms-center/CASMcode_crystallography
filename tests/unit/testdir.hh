#ifndef CASM_unittest_testdir
#define CASM_unittest_testdir

#include <filesystem>
#include <string>

#include "casm/global/filesystem.hh"

using namespace CASM;

namespace test {

/// Create a distinct temporary directory `./CASM_test_projects/tmp.<i>`
///
/// Removes itself upon destruction, unless `do_not_remove_on_destruction()` is
/// called. Use testing::Test::HasFatalFailure() to check for failures.
class TmpDir {
 public:
  TmpDir();

  /// Sets flag to remove the temp directory on destruction (set by default)
  void remove_on_destruction();

  /// Unsets flag to remove the temp directory on destruction
  void do_not_remove_on_destruction();

  ~TmpDir();

  fs::path path() const;

  operator fs::path() const;

 private:
  bool m_remove_on_destruction;
  fs::path m_path;
};

/// Store data files in "tests/unit/<module_name>/data/<file_name>"
fs::path data_dir(std::string module_name);

/// Store data files in "tests/unit/<module_name>/data/<file_name>"
fs::path data_file(std::string module_name, std::string file_name);

}  // namespace test

#endif
