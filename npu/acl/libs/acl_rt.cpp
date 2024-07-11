#include "acl/acl_rt.h"

ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext* context) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError
aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError
aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtResetOverflowStatus(aclrtStream stream) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtGetOverflowStatus(
    void* outputAddr,
    size_t outputSize,
    aclrtStream stream) {
  return 0;
}