#include <torch/csrc/dynamo/compiled_autograd.h>

#include "backend/npu/impl/framework/autograd/FunctionsManual.h"

#include "aten/CustomFunctions.h"

// ${generated_comment}

// The manual function definitions that used to be here are now in
// torch/csrc/autograd/FunctionsManual.cpp This speeds up re-compilation and
// allow to share these implementations so that they can be used for forward
// mode AD formulas as well.

using namespace at_npu::autograd::generated::details;
using namespace at_npu::native::custom_ops;
using at::IntArrayRef;
using at::Scalar;
using at::Tensor;
using at::TensorList;

namespace at_npu {
namespace autograd {
namespace generated {

${autograd_function_definitions}

} // namespace generated
} // namespace autograd
} // namespace at_npu
