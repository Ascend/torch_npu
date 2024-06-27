from typing import Any
from functools import lru_cache
import warnings

import torch
from torch._utils import _get_device_index as _torch_get_device_index

import torch_npu
import torch_npu._C
from torch_npu.utils.error_code import ErrCode, pta_error

__all__ = ["synchronize", "device_count", "can_device_access_peer", "set_device", "current_device", "get_device_name",
           "get_device_properties", "mem_get_info", "get_device_capability", "utilization", "device", "device_of",
           "_get_device_index"]


def synchronize(device=None):
    r"""Waits for all kernels in all streams on a NPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    torch_npu.npu._lazy_init()
    with torch_npu.npu.device(device):
        return torch_npu._C._npu_synchronize()


@lru_cache(maxsize=1)
def device_count():
    return torch_npu._C._npu_getDeviceCount()


def can_device_access_peer(device_id, peer_device_id):
    r"""Checks if peer access between two devices is possible.
    """
    device_id = _get_device_index(device_id, optional=True)
    peer_device_id = _get_device_index(peer_device_id, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    if peer_device_id < 0 or peer_device_id >= device_count():
        raise AssertionError("Invalid peer device id" + pta_error(ErrCode.VALUE))
    return torch_npu._C._npu_canDeviceAccessPeer(device_id, peer_device_id)


def set_device(device):
    device_id = _get_device_index(device, optional=True)
    if device_id >= 0:
        torch_npu._C._npu_setDevice(device_id)


def current_device():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDevice()


def get_device_name(device_name=None):
    device_prop = get_device_properties(device_name=device_name)
    return device_prop.name


def get_device_properties(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceProperties(device_id)


def mem_get_info(device=None):
    if device is None:
        device = torch_npu.npu.current_device()
    device_id = _get_device_index(device)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    device_prop = torch_npu._C._npu_getDeviceMemories(device_id)
    return device_prop.free_memory, device_prop.total_memory


def get_device_capability(device=None):
    r"""Query the minor and major data of device. Cann does not
    have a corresponding concept and is not supported. By default, it returns None
    """
    warnings.warn("torch.npu.get_device_capability isn't implemented!")
    return None


def utilization(device=None):
    r"""Query the comprehensive utilization rate of device
    """
    device_id = _get_device_index(device, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceUtilizationRate(device_id)


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch_npu._C._npu_getDevice()
        if self.prev_idx != self.idx:
            torch_npu._C._npu_setDevice(self.idx)
        torch_npu.npu._lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch_npu._C._npu_setDevice(self.prev_idx)
        return False


def _get_device_index(device: Any, optional: bool = False,
                      allow_cpu: bool = False) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a NPU device. Note that for a NPU device without a specified index,
    i.e., ``torch.device('npu')``, this will return the current default NPU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default NPU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ['npu', 'cpu']:
                raise ValueError('Expected a npu or cpu device, but got: {}'.format(device) + pta_error(ErrCode.VALUE))
        elif device.type != 'npu':
            raise ValueError('Expected a npu device, but got: {}'.format(device) + pta_error(ErrCode.VALUE))
    if not torch.jit.is_scripting():
        if isinstance(device, torch.npu.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_npu else -1
        super(device_of, self).__init__(idx)
