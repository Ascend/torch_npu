#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/macros/Macros.h>

namespace c10_backend::impl {

/**
 * All classes which inherit from PrivateUse1GuardImpl should be declared
 * 'final'.
 */
struct PrivateUse1GuardImpl : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  PrivateUse1GuardImpl() = default;

  explicit PrivateUse1GuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }

  c10::DeviceType type() const final {
    return c10::DeviceType::PrivateUse1;
  }
};

} // namespace c10_backend::impl
