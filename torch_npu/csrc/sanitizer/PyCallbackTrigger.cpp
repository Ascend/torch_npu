#include "torch_npu/csrc/sanitizer/PyCallbackTrigger.h"
#include <torch/csrc/utils/python_arg_parser.h>

namespace c10_npu {
namespace impl {

PyCallbackTrigger* getPyCallbackTrigger(const int mode) {
  return PyCallbackTrigger::instance(mode);
}

} // namespace impl
} // namespace c10_npu