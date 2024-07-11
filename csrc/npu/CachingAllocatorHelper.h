#include <c10/core/Device.h>
#include <c10/util/irange.h>
namespace c10_npu::NPUCachingAllocator {

//
// Struct and factory of ExpandableSegment
//

struct SegmentRange {
  char* ptr;
  size_t size;
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

class ExpandableSegment {
 public:
  virtual SegmentRange map(SegmentRange range) = 0;
  virtual SegmentRange unmap(SegmentRange range) = 0;
  virtual size_t size() const = 0;
  virtual char* ptr() const = 0;
  virtual ~ExpandableSegment() = default;
};

ExpandableSegment* createExpandableSegment(
    int device,
    void* stream,
    size_t size);

//
// event listener
//

class EventListener {
 public:
  virtual void before() {}
  virtual void after() {}
  virtual ~EventListener() = default;
};

std::unique_ptr<EventListener> createInsertEventListener(int device);

// Returns the current stream for the given device
void* getCurrentStream(c10::DeviceIndex);

} // namespace c10_npu::NPUCachingAllocator