import torch
import torch.nn as nn


class AWQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        assert w_bit in [4], "Only 4-bit are supported for now."
        assert group_size in [128], "Only 128 group size is supported for now."

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.split_k_iters = 8
        pack_num = 32 // self.w_bit  # 8

        n_group = in_features // group_size

        assert self.in_features % self.group_size == 0
        assert out_features % pack_num == 0

        def ceil_div(a, b):
            return (a + b - 1) // b

        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features // pack_num), dtype=torch.int32, device=dev
            ),
        )

        self.register_buffer(
            "qzeros",
            torch.zeros(
                (out_features, ceil_div(n_group, pack_num)),
                dtype=torch.int32,
                device=dev,
            ),
        )

        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    out_features,
                    ceil_div(n_group, pack_num) * pack_num,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None
