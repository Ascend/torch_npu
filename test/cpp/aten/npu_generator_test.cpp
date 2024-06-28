#include <gtest/gtest.h>
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

using namespace at_npu;

TEST(NPUGeneratorTest, TestSingletonDefaultGenerator) {
  auto gen = detail::getDefaultNPUGenerator();
  auto other = detail::getDefaultNPUGenerator();
  EXPECT_EQ(gen, other);
}

TEST(NPUGeneratorTest, TestCloning) {
  auto gen1 = detail::createNPUGenerator();
  auto npu_gen1 = at::check_generator<NPUGeneratorImpl>(gen1);

  auto gen2 = detail::createNPUGenerator();
  gen2 = gen1.clone();
  auto npu_gen2 = at::check_generator<NPUGeneratorImpl>(gen2);

  EXPECT_EQ(npu_gen1->current_seed(), npu_gen2->current_seed());
  EXPECT_EQ(
      npu_gen1->philox_offset_per_thread(),
      npu_gen2->philox_offset_per_thread());
}
