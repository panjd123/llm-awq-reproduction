import torch
from torch import nn
from my_awq.quantize.quantizer import pseudo_quantize_tensor
from awq.quantize.qmodule import WQLinear
from my_awq.qmodule.awqlinear import AWQLinear
from timeit import default_timer as timer


def tester(ic=4096, oc=4096, nloop=10, nbatch=5):

    weight = torch.randn(oc, ic).cuda().half()
    x = torch.randn(nbatch, ic).cuda().half()

    linear = nn.Linear(ic, oc).half().cuda()
    linear.weight.data = weight

    qweight, qscales, qzeros = pseudo_quantize_tensor(weight, get_scale_zp=True)
    linear.weight.data = qweight

    my_linear = AWQLinear.from_linear(
        linear, q_bit=4, group_size=128, scales=qscales, zeros=qzeros
    )

    awq_linear = WQLinear.from_linear(
        linear, w_bit=4, group_size=128, scales=qscales, zeros=qzeros
    )

    with torch.no_grad():
        tic0 = timer()
        for i in range(nloop):
            y = linear(x)
        tic1 = timer()
        for i in range(nloop):
            my_y = my_linear(x)
        tic2 = timer()
        for i in range(nloop):
            awq_y = awq_linear(x)
        tic3 = timer()

    y = y.cpu().float()
    my_y = my_y.cpu().float()
    awq_y = awq_y.cpu().float()
    my_loss = (y - my_y).abs().mean().item()
    awq_loss = (y - awq_y).abs().mean().item()

    # print(f"my_loss: {my_loss}")
    # print(f"awq_loss: {awq_loss}")

    t1 = tic1 - tic0
    t2 = tic2 - tic1
    t3 = tic3 - tic2

    print(f"linear time: {t1} (1x)")
    print(f"my_linear time: {t2} ({t1 / t2:.2f}x)")
    print(f"awq_linear time: {t3} ({t1 / t3:.2f}x)")

    if my_loss - awq_loss > 1e-3:
        print("Compute error!")
        print(qweight[0])
        print(qscales[0])
        print(qzeros[0])
        print(f"my_y: {my_y[0, 0]}")
        print(f"awq_y: {awq_y[0, 0]}")
        raise ValueError("Compute error!")
    else:
        print("Compute correct!")


if __name__ == "__main__":
    nloop = 10
    tester(256, 512, nloop, 4)
    tester(1024, 1024, nloop, 4)
    tester(4096, 4096, nloop, 4)

    tester(256, 512, nloop, 64)
    tester(1024, 1024, nloop, 64)
    tester(4096, 4096, nloop, 64)

    tester(256, 512, nloop, 512)
    tester(1024, 1024, nloop, 512)
    tester(4096, 4096, nloop, 512)
