#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include "backend/npu/impl/acl/include/acl/acl_base.h"
#include "backend/npu/impl/acl/include/acl/acl_rt.h"
#include "backend/npu/NPUCachingAllocator.h"
#include "backend/npu/impl/core/npu_log.h"

namespace c10_npu::NPUCachingAllocator {
struct NPUExpandableSegment : public ExpandableSegment {
  NPUExpandableSegment(int device, void* stream, size_t size)
      : device_(device),
        stream_(stream),
        max_handles_(0),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size) {
    size_t device_free;
    size_t device_total;
    NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
    // we allocate enough address space for 1 1/8 the total memory on the NPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(device_total + device_total / 8);
    NPU_CHECK_ERROR(c10_npu::acl::AclrtReserveMemAddress(
        &ptr_, segment_size_ * max_handles_, 0, NULL, 1));
    ASCEND_LOGD(
        "NPUCachingAllocator malloc by Aclr_tReserveMemAddress: size=%zu",
        segment_size_ * max_handles_);
  }
  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  SegmentRange map(SegmentRange range) override {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(
        ptr() + begin * segment_size_ == range.ptr, PTA_ERROR(ErrCode::PTR));
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }
    while (end > handles_.size()) {
      handles_.emplace_back(c10::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i), PTA_ERROR(ErrCode::VALUE));
      aclrtDrvMemHandle handle = nullptr;
      aclrtPhysicalMemProp prop = {};
      prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
      prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
      prop.memAttr = ACL_HBM_MEM_HUGE;
      prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
      prop.reserve = 0;
      auto status =
          c10_npu::acl::AclrtMallocPhysical(&handle, segment_size_, &prop, 0);
      if (status == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        for (auto j : c10::irange(begin, i)) {
          auto h = handles_.at(j).value();
          handles_.at(j) = c10::nullopt;
          NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h));
        }
        trimHandles();
        return rangeFromHandles(begin, begin);
      }
      NPU_CHECK_ERROR(status, "aclr_tMallocPhysical");
      handles_.at(i) = handle;
    }
    for (auto i : c10::irange(begin, end)) {
      NPU_CHECK_ERROR(c10_npu::acl::AclrtMapMem(
          ptr_ + i * segment_size_,
          segment_size_,
          0,
          handles_.at(i).value(),
          0));
    }
    ASCEND_LOGD("NPUCachingAllocator map: segment_size=%zu", segment_size_);
    return rangeFromHandles(begin, end);
  }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) override {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char* ptr() const override {
    return (char*)ptr_;
  }

  size_t size() const override {
    return max_handles_ * segment_size_;
  }

  ~NPUExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr_));
    ASCEND_LOGD("NPUCachingAllocator free by Aclr_tReleaseMemAddress");
  }

 private:
  void unmapHandles(size_t begin, size_t end) {
    // note: unlike aclr_tFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::npu::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    NPU_CHECK_ERROR(aclrtSynchronizeStream(stream_));
    for (auto i : c10::irange(begin, end)) {
      aclrtDrvMemHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      NPU_CHECK_ERROR(c10_npu::acl::AclrtUnmapMem(ptr_ + segment_size_ * i));
      NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h));
    }
    ASCEND_LOGD("NPUCachingAllocator unmap: segment_size=%zu", segment_size_);
    trimHandles();
  }

  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }

  void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
    auto start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }

  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }

  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }

  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }

  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }

  int device_;
  void* stream_;
  void* ptr_{};
  size_t max_handles_;
  size_t segment_size_;
  std::vector<c10::optional<aclrtDrvMemHandle>> handles_;
};
} // namespace c10_npu::NPUCachingAllocator
