from ._module import _apply_module_patch
from .tensor_methods import _add_tensor_methods
from .storage import _add_storage_methods
from .combine_tensors import npu_combine_tensors, get_part_combined_tensor, is_combined_tensor_valid
from .npu_intercept import _cann_package_check, _add_intercept_methods
from .collect_env import _add_collect_env_methods
from ._optim import add_optim_method
from .utils import print_error_log, print_warn_log, print_info_log

__all__ = ["npu_combine_tensors", "get_part_combined_tensor", "is_combined_tensor_valid"]
