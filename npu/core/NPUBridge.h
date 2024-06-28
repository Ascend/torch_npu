#pragma once
#include <c10/core/StorageImpl.h>
#include "npu/core/NPUStorageImpl.h"
#include "npu/core/NPUTensorImpl.h"

namespace torch_npu {

class NPUBridge {
 public:
  // at::tensor to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(const at::Tensor& tensor);

  // c10::StorageImpl to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(c10::StorageImpl* storageImpl);

  // c10::Storage to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(c10::Storage&& storage);

  // tensor to NPUStorageDesc
  static NPUStorageDesc& GetNpuStorageImplDesc(const at::Tensor& tensor);

  // tensor to NPUTensorImpl
  static NPUTensorImpl* GetNpuTensorImpl(const at::Tensor& tensor);
};
} // namespace torch_npu
