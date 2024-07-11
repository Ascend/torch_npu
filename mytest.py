
if __name__ == '__main__':
    import torch
    import torch_npu

    device = torch.device('npu')
    print(torch.ones(1, 2, device=device))
