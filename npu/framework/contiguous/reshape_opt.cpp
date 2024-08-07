#include "csrc/aten/generated/CustomFunctions.h"
#include "npu/framework/contiguous/ReshapeOpt.h"

namespace at_npu {
namespace native {

class ReshapeContiguousOpt : public ContiguousOpt {
 public:
  bool Optimizer(
      at::Tensor& result,
      const at::Tensor& src,
      const ContiguousTensorDesc& src_desc) override {
    ContiguousTensorDesc result_desc =
        TransContiguous::GetTensorDescInfo(result);
    if (check_reshape_match(result_desc, src_desc)) {
      RECORD_FUNCTION("contiguous_d_Reshape", std::vector<c10::IValue>({src}));
      custom_ops::npu_reshape_out(src, src.sizes(), false, result);
      return true;
    }
    return false;
  }

  bool CanOptimizer(const ContiguousTensorDesc& src_desc) override {
    return check_reshape_match(src_desc);
  }
}; // class ReshapeContiguousOpt

REGISTER_COPY_OPT(reshape, ReshapeContiguousOpt)

} // namespace native
} // namespace at_npu
