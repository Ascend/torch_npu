#include <cstddef>
#include <cstdint>

namespace c10_npu {
// Frees memory on the device. e.g. cudaFree
int memFree(void* devPtr);

// Allocate memory on the device. e.g. cudaMalloc
int memAlloc(void** devPtr, size_t size);

// Gets free and total device memory. e.g. cudaMemGetInfo
int memGetInfo(size_t* free, size_t* total);

// Free an address range reservation. e.g. cuMemAddressFree
int memAddressFree(void* ptr, size_t size);

// Allocate an address range reservation. e.g. cuMemAddressReserve
int memAddressReserve(
    void** virPtr,
    size_t size,
    size_t alignment,
    void* expectPtr,
    uint64_t flags);
} // namespace c10_npu
