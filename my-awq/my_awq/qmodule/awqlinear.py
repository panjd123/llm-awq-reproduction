import torch
import torch.nn as nn


def ceil_div(a, b):
    return (a + b - 1) // b


def align_up(a, b):
    return ((a + b - 1) // b) * b


class AWQLinear(nn.Module):
    def __init__(self, q_bit, group_size, in_features, out_features, bias, device):
        super().__init__()

        assert q_bit in [4], "Only 4-bit are supported for now."
        assert group_size in [128], "Only 128 group size is supported for now."

        self.in_features = in_features
        self.out_features = out_features
        self.q_bit = q_bit
        self.group_size = group_size
        self.split_k_iters = 8
        pack_num = 32 // self.q_bit  # 8
        self.pack_num = pack_num

        assert in_features % group_size == 0
        assert in_features % pack_num == 0

        n_group = in_features // group_size

        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features // pack_num),
                dtype=torch.int32,
                device=device,
            ),
        )

        self.register_buffer(
            "qzeros",
            torch.zeros(
                (out_features, ceil_div(n_group, pack_num)),
                dtype=torch.int32,
                device=device,
            ),
        )

        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    out_features,
                    align_up(n_group, pack_num),
                ),
                dtype=torch.float16,
                device=device,
            ),
        )

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=device)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, q_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        device = linear.weight.device
        awq_linear = cls(
            q_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device,
        )
        if init_only:
            return awq_linear

        scales = scales.to(device=device)
        zeros = zeros.to(device=device)

        ic = awq_linear.in_features
        # oc = awq_linear.out_features
        pack_num = awq_linear.pack_num
        n_group = ic // group_size

        # scales
        padding_scales = torch.zeros_like(awq_linear.scales, device=device)
        padding_scales[:, : scales.shape[1]] = scales
        awq_linear.scales = padding_scales

        # bias
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().to(torch.float16)

        # qweight
        weight = linear.weight.data  # (oc, ic)
        int_weight = []

        print(weight.device, padding_scales.device, zeros.device)

        for i in range(ic):  # shape[1]
            col = (
                weight[:, i] / padding_scales[:, i // group_size]
                + zeros[:, i // group_size]
            )
            int_weight.append(col.to(torch.int32)[:, None])
        int_weight = torch.cat(int_weight, dim=1)

        pack_weight = torch.zeros_like(awq_linear.qweight, device=device)

        for i in range(ic // pack_num):  # shape[1]
            for j in range(pack_num):
                pack_weight[:, i] |= int_weight[:, i * pack_num + j] << (j * q_bit)

        awq_linear.qweight = pack_weight

        # zeros
        zeros = zeros.to(torch.int32)
        qzeros = torch.zeros_like(awq_linear.qzeros, device=device)
        print(qzeros.shape)
        print(n_group, pack_num)

        for i in range(ceil_div(n_group, pack_num)):
            for j in range(pack_num):
                if i * pack_num + j >= n_group:
                    break
                qzeros[:, i] |= zeros[:, i * pack_num + j] << (j * q_bit)

        awq_linear.qzeros = qzeros

        return awq_linear

    @torch.no_grad()  # for now
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        inputs = x.reshape(-1, x.shape[-1])
        out = torch.zeros(
            (inputs.shape[0], self.out_features), device=x.device, dtype=torch.float16
        )  # TODO
        if self.bias is not None:
            out += self.bias
        return out.reshape(out_shape)
