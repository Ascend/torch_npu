#include <torch/csrc/utils/tensor_new.h>

#include "npu/core/npu/NPUMacros.h"
#include "npu/core/npu/NPUFunctions.h"

namespace torch_npu {
namespace utils {

// Initializes the Python tensor type objects: torch.npu.FloatTensor,
// torch.npu.DoubleTensor, etc. and binds them in their containing modules.
void _initialize_python_bindings();

TORCH_NPU_API PyMethodDef* npu_extension_functions();

}
}
