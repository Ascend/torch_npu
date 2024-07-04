#include <torch/csrc/utils/python_stub.h>
#include "backend/npu/impl/core/NPUException.h"
#include "backend/npu/impl/core/NPUMacros.h"

namespace torch_npu {
namespace autocast {

TORCH_NPU_API PyMethodDef* autocast_mode_functions();

}
} // namespace torch_npu
