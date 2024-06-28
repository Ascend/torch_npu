#ifndef PROFILER_NPU_INC
#define PROFILER_NPU_INC

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <string>
#include <unordered_map>

#include "npu/core/npu/NPUException.h"

namespace torch_npu {
namespace profiler {

std::unordered_map<std::string, c10::IValue> saveExtraArgs(
    const at::RecordFunction& fn);

uint64_t computeFlops(
    const std::string& op_name,
    const std::unordered_map<std::string, c10::IValue>& extra_args);

class NPURecordFunction {
 public:
  NPURecordFunction(bool enable_ = false) : enable(enable_) {
    if (NPURecordFunction::use_npu_simple) {
      at::enableRecordFunction(enable);
    }
  }

  ~NPURecordFunction() {
    if (NPURecordFunction::use_npu_simple) {
      at::enableRecordFunction(!enable);
    }
  }
  bool enable = false;
  static bool use_npu_simple;
};

} // namespace profiler
} // namespace torch_npu

#endif // PROFILER_NPU_INC
