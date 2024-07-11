#include <gtest/gtest.h>
#include "csrc/npu/NPUGeneratorImpl.h"

TEST(NPUGeneratorImpl, TestSingletonDefaultGenerator) {
  auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto other = at_npu::detail::getDefaultNPUGenerator();
  EXPECT_EQ(gen, other);
}

TEST(NPUGeneratorImpl, TestCloning) {
  auto gen1 = at_npu::detail::createNPUGenerator();
  auto npu_gen1 = at::check_generator<at_npu::NPUGeneratorImpl>(gen1);

  auto gen2 = at_npu::detail::createNPUGenerator();
  gen2 = gen1.clone();
  auto npu_gen2 = at::check_generator<at_npu::NPUGeneratorImpl>(gen2);

  EXPECT_EQ(npu_gen1->current_seed(), npu_gen2->current_seed());
  EXPECT_EQ(
      npu_gen1->philox_offset_per_thread(),
      npu_gen2->philox_offset_per_thread());
}

TEST(NPUGeneratorImpl, TestGetSetCurrentSeed) {
  auto gen = at_npu::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<at_npu::NPUGeneratorImpl>(gen);
  npu_gen->set_current_seed(10);
  EXPECT_EQ(npu_gen->current_seed(), 10);
}

TEST(NPUGeneratorImpl, TestDeviceType) {
  EXPECT_EQ(
      at_npu::NPUGeneratorImpl::device_type(), c10::DeviceType::PrivateUse1);
}

TEST(NPUGeneratorImpl, TestGetSetOffset) {
  auto gen = at_npu::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<at_npu::NPUGeneratorImpl>(gen);
  npu_gen->set_offset(100);
  EXPECT_EQ(npu_gen->get_offset(), 100);
}

TEST(NPUGeneratorImpl, TestGetSetPhiloxOffset) {
  auto gen = at_npu::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<at_npu::NPUGeneratorImpl>(gen);
  npu_gen->set_philox_offset_per_thread(200);
  EXPECT_EQ(npu_gen->philox_offset_per_thread(), 200);
}