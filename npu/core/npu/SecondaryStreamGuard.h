#pragma once

#include "npu/core/npu/NPUEvent.h"
#include "npu/core/npu/NPUGuard.h"
#include "npu/core/npu/NPUStream.h"

namespace c10_npu {
struct SecondaryStreamGuard {
  explicit SecondaryStreamGuard() = delete;
  explicit SecondaryStreamGuard(c10::Stream stream) : guard_(stream) {};

  ~SecondaryStreamGuard();

 private:
  c10_npu::NPUStreamGuard guard_;
};
} // namespace c10_npu
