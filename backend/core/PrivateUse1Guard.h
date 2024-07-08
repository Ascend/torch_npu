#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "backend/core/impl/PrivateUse1GuardImpl.h"

namespace c10_backend {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for PrivateUse1.  It accepts
/// integer indices (interpreting them as PrivateUse1 devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// cudaSetDevice/cudaGetDevice calls); however, it can only be used
/// from code that links against PrivateUse1 directly.
template <typename T>
struct PrivateUse1Guard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit PrivateUse1Guard() = delete;

  /// Set the current PrivateUse1 device to the passed device index.
  explicit PrivateUse1Guard(c10::DeviceIndex device_index)
      : guard_(device_index) {}

  /// Sets the current PrivateUse1 device to the passed device.  Errors if the
  /// passed device is not a PrivateUse1 device.
  explicit PrivateUse1Guard(c10::Device device) : guard_(device) {}

  // Copy is not allowed
  PrivateUse1Guard(const PrivateUse1Guard&) = delete;

  PrivateUse1Guard& operator=(const PrivateUse1Guard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  PrivateUse1Guard(PrivateUse1Guard&& other) = delete;

  PrivateUse1Guard& operator=(PrivateUse1Guard&& other) = delete;
  /// Sets the PrivateUse1 device to the given device.  Errors if the given
  /// device is not a PrivateUse1 device.
  void set_device(c10::Device device) {
    guard_.set_device(device);
  }

  /// Sets the PrivateUse1 device to the given device.  Errors if the given
  /// device is not a PrivateUse1 device.  (This method is provided for
  /// uniformity with DeviceGuard).
  void reset_device(c10::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the PrivateUse1 device to the given device index.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  c10::Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  c10::Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<T> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for PrivateUse1.  See
/// PrivateUse1Guard for when you can use this.
template <typename T>
struct OptionalPrivateUse1Guard {
  /// Create an uninitialized OptionalPrivateUse1Guard.
  explicit OptionalPrivateUse1Guard() : guard_() {}

  /// Set the current PrivateUse1 device to the passed Device, if it is not
  /// nullopt.
  explicit OptionalPrivateUse1Guard(std::optional<c10::Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current PrivateUse1 device to the passed device index, if it is
  /// not nullopt
  explicit OptionalPrivateUse1Guard(
      std::optional<c10::DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalPrivateUse1Guard(const OptionalPrivateUse1Guard&) = delete;

  OptionalPrivateUse1Guard& operator=(const OptionalPrivateUse1Guard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalPrivateUse1Guard(OptionalPrivateUse1Guard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalPrivateUse1Guard& operator=(OptionalPrivateUse1Guard&& other) =
      delete;

  /// Sets the PrivateUse1 device to the given device, initializing the guard if
  /// it is not already initialized.  Errors if the given device is not a
  /// PrivateUse1 device.
  void set_device(c10::Device device) {
    guard_.set_device(device);
  }

  /// Sets the PrivateUse1 device to the given device, initializing the guard if
  /// it is not already initialized.  Errors if the given device is not a
  /// PrivateUse1 device. (This method is provided for uniformity with
  /// OptionalDeviceGuard).
  void reset_device(c10::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the PrivateUse1 device to the given device index, initializing the
  /// guard if it is not already initialized.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  std::optional<c10::Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  std::optional<c10::Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original PrivateUse1 device, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<T> guard_;
};

} // namespace c10_backend
