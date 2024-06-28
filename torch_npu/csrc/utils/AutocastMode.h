#include <torch/csrc/utils/python_stub.h>
#include "npu/core/npu/NPUException.h"
#include "npu/core/npu/NPUMacros.h"

namespace torch_npu {
namespace autocast {

TORCH_NPU_API PyMethodDef* autocast_mode_functions();

}
} // namespace torch_npu
