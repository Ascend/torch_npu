#pragma once

#include <npu/acl/include/acl/acl.h>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "c10/macros/Export.h"
#include "csrc/npu/NPUEventManager.h"
#include "npu/core/NPUMacros.h"
#include "npu/core/npu_log.h"
#define NpuSysStatus c10_npu::NpuSysCtrl::SysStatus

namespace c10_npu {
using ReleaseFn = std::function<void()>;

enum class ReleasePriority : uint8_t {
  PriorityFirst = 0,
  PriorityMiddle = 5,
  PriorityLast = 10
};

class NpuSysCtrl {
 public:
  ~NpuSysCtrl() = default;

  enum SysStatus {
    INIT_SUCC = 0,
    INIT_ALREADY,
    INIT_FAILED,
    CREATE_SESS_SUCC,
    CREATE_SESS_FAILED,
    FINALIZE_SUCC,
    FINALIZE_FAILED,
  };

  // Get NpuSysCtrl singleton instance
  C10_NPU_API static NpuSysCtrl& GetInstance();

  // Environment Initialize, return SysStatus
  SysStatus Initialize(int device_id = -1);

  // Change current device from pre_device to device
  SysStatus ExchangeDevice(int pre_device, int device);

  // Init backwards thread
  SysStatus BackwardsInit();

  // Set overflow switch
  SysStatus OverflowSwitchEnable();

  // Environment Finalize, return SysStatus
  C10_NPU_API SysStatus Finalize();

  // Get Init_flag
  C10_NPU_API bool GetInitFlag();

  int InitializedDeviceID();

  // Register fn to be called during stage of exit and
  // the callability of fn is guaranteed by the caller.
  void RegisterReleaseFn(
      ReleaseFn release_fn,
      ReleasePriority priority = ReleasePriority::PriorityMiddle);

 private:
  NpuSysCtrl();

 private:
  bool repeat_init_acl_flag_;
  bool init_flag_;
  int device_id_;
  std::map<ReleasePriority, std::vector<ReleaseFn>> release_fn_;
};

aclError SetCurrentDevice();

} // namespace c10_npu
