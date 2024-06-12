import torch
import my_awq_kernels

a = torch.tensor([1, 2, 3]).cuda().half()
b = torch.tensor([4, 5, 6]).cuda().half()
c = my_awq_kernels.myadd_cuda(a, b)
if (c.cpu().numpy() == [5, 7, 9]).all():
    print("Success!")
else:
    raise RuntimeError("Error!")
