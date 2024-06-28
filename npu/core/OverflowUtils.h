#pragma once

#include "aten/NPUNativeFunctions.h"

namespace torch_npu {
namespace utils {

class OverflowUtil {
 public:
  ~OverflowUtil();

  static OverflowUtil* GetInstance() {
    static OverflowUtil instance;
    return &instance;
  }

  void EnableOverflowNpu();
  bool CheckOverflowNpu();
  void ClearOverflowNpu();

 private:
  OverflowUtil();
  bool hasOverflow = false;
};

} // namespace utils
} // namespace torch_npu
