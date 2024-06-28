#ifndef __TORCH_NPU_MSPROFILERINTERFACE__
#define __TORCH_NPU_MSPROFILERINTERFACE__

#include <npu/acl/include/acl/acl_msprof.h>
#include "npu/core/npu/NPUException.h"

namespace at_npu {
namespace native {

aclError AclprofSetConfig(
    aclprofConfigType configType,
    const char* config,
    size_t configLength);

aclError AclprofGetSupportedFeatures(size_t* featuresSize, void** featuresData);

aclError AclProfilingMarkEx(const char* msg, size_t msgLen, aclrtStream stream);
} // namespace native
} // namespace at_npu

#endif // __TORCH_NPU_MSPROFILERINTERFACE__
