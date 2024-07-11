#include "csrc/npu/Memory.h"
#include "npu/acl/include/acl/acl_rt.h"
#include "npu/core/interface/AclInterface.h"

namespace c10_npu {
int memFree(void* devPtr) {
  return aclrtFree(devPtr);
}

int memAlloc(void** devPtr, size_t size) {
  return aclrtMalloc(
      devPtr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
}

int memGetInfo(size_t* free, size_t* total) {
  return aclrtGetMemInfo(ACL_HBM_MEM, free, total);
}

int memAddressFree(void* ptr, size_t size) {
  return acl::AclrtReleaseMemAddress(ptr);
}

int memAddressReserve(
    void** virPtr,
    size_t size,
    size_t alignment,
    void* expectPtr,
    uint64_t flags) {
  return acl::AclrtReserveMemAddress(virPtr, size, alignment, expectPtr, flags);
}
} // namespace c10_npu
