#include <cstddef>
#include <cstdint>

namespace c10_npu {
// e.g. cudaFree
int memFree(void* devPtr);

// e.g. cudaMalloc
int memAlloc(void** devPtr, size_t size);

// e.g. cudaMemGetInfo
int memGetInfo(size_t* free, size_t* total);

// e.g. cuMemAddressFree
int memAddressFree(void* ptr, size_t size);

// e.g. cuMemAddressReserve
int memAddressReserve(
    void** virPtr,
    size_t size,
    size_t alignment,
    void* expectPtr,
    uint64_t flags);
} // namespace c10_npu
