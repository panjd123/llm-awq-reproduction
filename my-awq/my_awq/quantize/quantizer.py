import torch


@torch.no_grad()
def pseudo_quantize_tensor(
    w, q_bit=4, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    """
    scales = (max_val - min_val) / max_int

    zeros = -min_val / scales

    q = clamp(round(w / scales) + zeros)

    w = (q - zeros) * scales

    w: [oc, ic]

    scales, zeros: [oc, ic / group_size]
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**q_bit - 1
        min_int = 0
        q_scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / q_scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (q_bit - 1) - 1
        min_int = -(2 ** (q_bit - 1))
        q_scales = max_val / max_int
        zeros = 0

    assert torch.isnan(q_scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(q_scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(q_scales)
    else:
        w = (
            torch.clamp(torch.round(w / q_scales) + zeros, min_int, max_int) - zeros
        ) * q_scales

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, q_scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
