#include "aten/BaseGeneratorImpl.h"
#include <ATen/EmptyTensor.h>
#include <ATen/Utils.h>

namespace at {

BaseGeneratorImpl::BaseGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl{
          Device(DeviceType::PrivateUse1, device_index),
          DispatchKeySet(DispatchKey::PrivateUse1)} {}

void BaseGeneratorImpl::set_offset(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

uint64_t BaseGeneratorImpl::get_offset() const {
  return philox_offset_per_thread_;
}

uint64_t BaseGeneratorImpl::current_seed() const {
  return seed_;
}

void BaseGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

uint64_t BaseGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}
} // namespace at