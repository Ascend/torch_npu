import torch
import torch_npu

print(torch.npu.FloatTensor.is_npu) # True
print(torch.npu.FloatTensor.dtype)  # torch.float32
print(torch.npu.FloatTensor.layout) # torch.strided


