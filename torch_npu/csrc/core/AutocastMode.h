#include <torch/csrc/utils/python_stub.h>
#include "npu/core/NPUException.h"
#include "npu/core/NPUMacros.h"

namespace torch_npu {
namespace autocast {

TORCH_NPU_API PyMethodDef* autocast_mode_functions();

}
} // namespace torch_npu
