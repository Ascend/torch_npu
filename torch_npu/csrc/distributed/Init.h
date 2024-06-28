#pragma once

#include <torch/csrc/python_headers.h>
#include "npu/core/npu/NPUMacros.h"

namespace torch_npu {
namespace distributed {

TORCH_NPU_API PyMethodDef* python_functions();

} // namespace distributed
} // namespace torch_npu
