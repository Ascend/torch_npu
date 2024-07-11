#include <gtest/gtest.h>

int add(int a, int b) {
  return a + b;
}

TEST(DeviceTest, First) {
  EXPECT_EQ(-3, add(-2, -1));
  EXPECT_EQ(-2, add(1, -3));
}
