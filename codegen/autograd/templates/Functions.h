#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/TensorGeometry.h>
#include <ATen/core/functional.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>

#include <c10/core/SymIntArrayRef.h>

using namespace torch::autograd;

namespace at_npu {
namespace autograd {
namespace generated {

using at::ArrayRef;
using at::IntArrayRef;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorGeometry;
using at::Type;
using c10::fmap;
using c10::optional;

inline std::vector<Tensor> unpack_list(
    at::ArrayRef<SavedVariable> xs,
    std::shared_ptr<Node> saved_for = nullptr) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [&saved_for](const SavedVariable& x) {
    return static_cast<Tensor>(x.unpack(saved_for));
  });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(
    at::ArrayRef<SavedVariable> xs,
    std::shared_ptr<Node> saved_for = nullptr) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    auto var = v.unpack(saved_for);
    result.push_back(var.defined() ? c10::optional<Tensor>(var) : c10::nullopt);
  }
  return result;
}

struct TypeAndSize {
  TypeAndSize() : options(at::TensorOptions()) {}
  /* implicit */
  TypeAndSize(const Tensor& t) : sizes(t.sizes().vec()), options(t.options()) {}

  Tensor zeros() {
    return at::zeros(sizes, options);
  }

 private:
  std::vector<int64_t> sizes;
  at::TensorOptions options;
};

${autograd_function_declarations}

} // namespace generated
} // namespace autograd
} // namespace at_npu
