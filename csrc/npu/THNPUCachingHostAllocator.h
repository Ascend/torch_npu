#include <c10/core/Allocator.h>
#include <c10/util/SmallVector.h>

#include <npu/acl/include/acl/acl.h>
#include "npu/core/NPUException.h"
#include "npu/core/NPUMacros.h"
#include "csrc/npu/NPUStream.h"

c10::Allocator* getTHNPUCachingHostAllocator(void);

aclError THNPUCachingHostAllocator_recordEvent(
    void* ptr,
    c10_npu::NPUStream stream);

bool THNPUCachingHostAllocator_isPinndPtr(void* ptr);
// Releases cached pinned memory allocations via npuHostFree
TORCH_NPU_API void THNPUCachingHostAllocator_emptyCache(void);

c10::Allocator* getPinnedMemoryAllocator(void);
