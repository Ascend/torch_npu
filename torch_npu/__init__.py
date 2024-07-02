import atexit

from functools import wraps

import torch
import torch.utils

import torch_npu
import torch_npu._C
import torch_npu.npu
import torch_npu.optim
from torch_npu.utils.error_code import _except_handler, ErrCode, pta_error

from torch_npu.utils.exposed_api import public_npu_functions
from . import _op_plugin_docs
from .meta import _meta_registrations

del _op_plugin_docs


def _wrap_torch_error_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(
            f"torch.{func.__name__} is deprecated and will be removed in future version. "
            f"Use torch_npu.{func.__name__} instead." + pta_error(ErrCode.NOT_SUPPORT)
        )

    return wrapper


for name in dir(torch.ops.npu):
    if name.startswith("__") or name in ["_dir", "name"]:
        continue
    globals()[name] = getattr(torch.ops.npu, name)
    if name in public_npu_functions:
        __all__.append(name)
    setattr(torch, name, _wrap_torch_error_func(getattr(torch.ops.npu, name)))


torch.utils.rename_privateuse1_backend("npu")
# rename device name to 'npu' and register funcs
torch._register_device_module("npu", torch_npu.npu)
unsupported_dtype = [
    torch.quint8,
    torch.quint4x2,
    torch.quint2x4,
    torch.qint32,
    torch.qint8,
]
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
    unsupported_dtype=unsupported_dtype,
)

# Apply monkey-patches.
_except_handler.patch_excepthook()
# this must be placed at the end
supported_dtypes = [
    torch.uint8,
    torch.int8,
    torch.float64,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.float16,
    torch.bool,
    torch.bfloat16,
]
torch_npu._C.generate_tensor_types(supported_dtypes)


# NPU exit, need to synchronize devices
def _npu_shutdown():
    torch_npu._C._npu_shutdown()
    _except_handler.handle_exception()


# register npu shutdown hook on exit
atexit.register(_npu_shutdown)
