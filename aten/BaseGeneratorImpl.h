#pragma once

#include <ATen/core/Generator.h>

namespace at {
struct C10_API BaseGeneratorImpl : public GeneratorImpl {
  BaseGeneratorImpl(DeviceIndex device_index);

  BaseGeneratorImpl(const BaseGeneratorImpl& other) = delete;
  BaseGeneratorImpl(BaseGeneratorImpl&& other) = delete;
  BaseGeneratorImpl& operator=(const BaseGeneratorImpl& other) = delete;

  ~BaseGeneratorImpl() override = default;

  c10::intrusive_ptr<BaseGeneratorImpl> clone() const;

  virtual void set_current_seed(uint64_t seed) override;
  virtual void set_offset(uint64_t offset) override;
  virtual uint64_t get_offset() const override;
  virtual uint64_t current_seed() const override;
  virtual uint64_t seed() override;

 private:
  uint64_t philox_offset_per_thread_;
  uint64_t seed_ = default_rng_seed_val;
};

} // namespace at
