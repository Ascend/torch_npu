#ifndef PROFILER_INIT_INC
#define PROFILER_INIT_INC
#include "npu/core/npu/NPUMacros.h"

namespace torch_npu {
namespace profiler {
TORCH_NPU_API PyMethodDef* profiler_functions();
}
} // namespace torch_npu

#endif // PROFILER_INIT_INC
