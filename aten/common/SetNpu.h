#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

#include "aten/NPUNativeFunctions.h"
#include "aten/common/ResizeNpu.h"
#include "npu/core/NPUBridge.h"
#include "npu/core/NPUStorageImpl.h"
#include "npu/core/npu/NPUCachingAllocator.h"
#include "npu/framework/StorageDescHelper.h"

namespace at_npu {
namespace native {

at::Tensor set_tensor_with_storage_format(c10::Storage src);

} // namespace native
} // namespace at_npu
