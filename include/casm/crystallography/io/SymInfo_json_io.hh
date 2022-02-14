#ifndef CASM_xtal_SymInfo_json_io
#define CASM_xtal_SymInfo_json_io

#include "casm/casm_io/enum/json_io.hh"
#include "casm/crystallography/SymInfo.hh"
#include "casm/crystallography/io/SymInfo_stream_io.hh"
#include "casm/global/definitions.hh"

namespace CASM {
namespace xtal {
struct SymInfo;
struct SymInfoOptions;
}  // namespace xtal

template <typename T>
struct jsonConstructor;
class jsonParser;

ENUM_JSON_IO_DECL(xtal::symmetry_type)

jsonParser &to_json(const xtal::SymInfoOptions &opt, jsonParser &json);

/// \brief Read from JSON
void from_json(xtal::SymInfoOptions &opt, const jsonParser &json);

template <>
struct jsonConstructor<xtal::SymInfoOptions> {
  static xtal::SymInfoOptions from_json(const jsonParser &json);
};

/// \brief Adds to existing JSON object
void to_json(const xtal::SymInfo &info, jsonParser &json);

}  // namespace CASM

#endif
