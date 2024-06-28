#include "npu/core/NPUSerialization.h"
#include "aten/NPUNativeFunctions.h"
#include "aten/common/FormatCastHelper.h"
#include "npu/framework/FormatHelper.h"
#include "npu/framework/StorageDescHelper.h"
#include "npu/acl/include/acl/acl_base.h"

namespace torch_npu {

std::unordered_map<std::string, aclFormat> FORMAT_INFO = {
    {"NC1HWC0", ACL_FORMAT_NC1HWC0},
    {"ND", ACL_FORMAT_ND},
    {"NCHW", ACL_FORMAT_NCHW},
    {"NHWC", ACL_FORMAT_NHWC},
    {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
    {"FRACTAL_Z", ACL_FORMAT_FRACTAL_Z},
    {"NDHWC", ACL_FORMAT_NDHWC},
    {"NCDHW", ACL_FORMAT_NCDHW},
    {"NDC1HWC0", ACL_FORMAT_NDC1HWC0},
    {"FRACTAL_Z_3D", ACL_FRACTAL_Z_3D},
};

void npu_info_serialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& map) {
  at_npu::native::StorageDescHelper::GetDescForSerialization(t, map);
}

void npu_info_deserialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& map) {
  // Set the true stroage description
  at_npu::native::StorageDescHelper::SetDescForSerialization(t, map);

  auto str_to_aclFormat = [](std::string str) -> aclFormat {
    int start = 0;
    while (str[start++] != '/')
      ;
    return FORMAT_INFO[str.substr(start, str.size() - start)];
  };

  for (auto& m : map) {
    if (m.first.find("npu_format_") != std::string::npos) {
      aclFormat format = str_to_aclFormat(m.first);
      // The format cast is an operator,
      // so special handling is required for scenarios
      // where the leaf node tensor requires grad at the same time
      bool revert_flag = false;
      if (t.is_leaf() && t.requires_grad()) {
        revert_flag = true;
        t.set_requires_grad(false);
      }
      at_npu::native::NPUNativeFunctions::npu_format_cast_(
          const_cast<at::Tensor&>(t), format);
      if (revert_flag) {
        t.set_requires_grad(true);
      }
    }
  }
}

} // namespace torch_npu
