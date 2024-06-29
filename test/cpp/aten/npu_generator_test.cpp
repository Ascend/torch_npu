#include <gtest/gtest.h>
#include "aten/NPUGeneratorImpl.h"

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