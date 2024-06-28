#include "aten/NPUNativeFunctions.h"
#include "npu/core/NPUBridge.h"
#include "npu/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

int64_t NPUNativeFunctions::get_storage_size(const at::Tensor& self) {
  torch_npu::utils::torch_check_npu(self);
  auto sizes =
      torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
  int64_t n = 1;
  for (auto s : sizes) {
    n *= s;
  }
  return n;
}

} // namespace native
} // namespace at_npu
