#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include "backend/core/PrivateUse1Guard.h"
#include "backend/npu/NPUGuardImpl.h"
#include "backend/npu/impl/core/NPUMacros.h"

#include <cstddef>

namespace c10_npu {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for NPU.  It accepts
/// integer indices (interpreting them as NPU devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// NPUSetDevice/NPUGetDevice calls); however, it can only be used
/// from code that links against NPU directly.
struct NPUGuard : public c10_backend::PrivateUse1Guard<impl::NPUGuardImpl> {
  using PrivateUse1Guard = c10_backend::PrivateUse1Guard<impl::NPUGuardImpl>;
  using PrivateUse1Guard::PrivateUse1Guard;

  // Copy is not allowed
  NPUGuard(const NPUGuard&) = delete;
  NPUGuard& operator=(const NPUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  NPUGuard(NPUGuard&& other) = delete;
  NPUGuard& operator=(NPUGuard&& other) = delete;
};

/// A variant of OptionalDeviceGuard that is specialized for NPU.  See
/// NPUGuard for when you can use this.
struct OptionalNPUGuard
    : public c10_backend::OptionalPrivateUse1Guard<impl::NPUGuardImpl> {
  using OptionalPrivateUse1Guard =
      c10_backend::OptionalPrivateUse1Guard<impl::NPUGuardImpl>;
  using OptionalPrivateUse1Guard::OptionalPrivateUse1Guard;

  // Copy is not allowed
  OptionalNPUGuard(const OptionalNPUGuard&) = delete;
  OptionalNPUGuard& operator=(const OptionalNPUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalNPUGuard(OptionalNPUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalNPUGuard& operator=(OptionalNPUGuard&& other) = delete;
};

/// A variant of StreamGuard that is specialized for NPU.  See NPUGuard
/// for when you can use this.
struct NPUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit NPUStreamGuard() = delete;

  /// Set the current NPU device to the device associated with the passed
  /// stream, and set the current NPU stream on that device to the passed
  /// stream. Errors if the Stream is not a NPU stream.
  explicit NPUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  NPUStreamGuard(const NPUStreamGuard&) = delete;
  NPUStreamGuard& operator=(const NPUStreamGuard&) = delete;

  /// Move is disallowed, as NPUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  NPUStreamGuard(NPUStreamGuard&& other) = delete;
  NPUStreamGuard& operator=(NPUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a NPU stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on NPU, use NPUMultiStreamGuard instead.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the NPU stream that was set at the time the guard was constructed.
  NPUStream original_stream() const {
    return NPUStream(NPUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent NPU stream that was set using this device guard,
  /// either from construction, or via set_stream.
  NPUStream current_stream() const {
    return NPUStream(NPUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent NPU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  c10::Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the NPU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  c10::Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<c10_npu::impl::NPUGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for NPU.  See NPUGuard
/// for when you can use this.
struct OptionalNPUStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalNPUStreamGuard() : guard_() {}

  /// Set the current NPU device to the device associated with the passed
  /// stream, and set the current NPU stream on that device to the passed
  /// stream. Errors if the Stream is not a NPU stream.
  explicit OptionalNPUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalNPUStreamGuard(c10::optional<c10::Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalNPUStreamGuard(const OptionalNPUStreamGuard&) = delete;
  OptionalNPUStreamGuard& operator=(const OptionalNPUStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalNPUStreamGuard(OptionalNPUStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalNPUStreamGuard& operator=(OptionalNPUStreamGuard&& other) = delete;

  /// Resets the currently set NPU stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the NPU stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  c10::optional<NPUStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return c10::make_optional(NPUStream(NPUStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Returns the most recent NPU stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  c10::optional<NPUStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return c10::make_optional(NPUStream(NPUStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Restore the original NPU device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<c10_npu::impl::NPUGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for NPU.
struct NPUMultiStreamGuard {
  explicit NPUMultiStreamGuard(at::ArrayRef<NPUStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  NPUMultiStreamGuard(const NPUMultiStreamGuard&) = delete;
  NPUMultiStreamGuard& operator=(const NPUMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  NPUMultiStreamGuard(NPUMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  NPUMultiStreamGuard& operator=(NPUMultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<c10_npu::impl::NPUGuardImpl> guard_;

  static std::vector<c10::Stream> unwrapStreams(
      at::ArrayRef<NPUStream> NPUStreams) {
    std::vector<c10::Stream> streams;
    streams.reserve(NPUStreams.size());
    for (const NPUStream& NPUStream : NPUStreams) {
      streams.push_back(NPUStream);
    }
    return streams;
  }
};

} // namespace c10_npu
