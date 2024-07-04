#include <torch/csrc/utils/tensor_new.h>

#include "backend/npu/NPUFunctions.h"
#include "backend/npu/impl/core/NPUMacros.h"

namespace torch_npu {
namespace utils {

// Initializes the Python tensor type objects: torch.npu.FloatTensor,
// torch.npu.DoubleTensor, etc. and binds them in their containing modules.
void initialize_python_bindings(std::vector<at::ScalarType>& scalar_types);

TORCH_NPU_API PyMethodDef* npu_extension_functions();

} // namespace utils
} // namespace torch_npu
