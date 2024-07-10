#pragma once
#include <c10/core/TensorOptions.h>
#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "npu/core/sys_ctrl/npu_sys_ctrl.h"

namespace torch_npu {
namespace utils {

void npu_lazy_init();

void npu_set_run_yet_variable_to_false();

} // namespace utils
} // namespace torch_npu
