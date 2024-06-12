import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import tqdm

from ..utils.utils import (
    get_op_by_name,
    clear_memory,
    set_op_by_name,
    get_device,
)
from .awq_module_extract import get_layers, get_named_linears
from .quantizer import pseudo_quantize_tensor
from ..qmodule.awqlinear import AWQLinear


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))


def apply_awq_scale(module, scales_list, input_feat_dict=None):
    for prev_layer_name, layer_names, scales in tqdm.tqdm(
        scales_list, desc="apply awq scale"
    ):
        prev_op = get_op_by_name(module, prev_layer_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)):
            scale_ln_fcs(prev_op, layers, scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    q_config,
):

    layers = get_layers(model)
    for i in tqdm.tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            # m.weight.data = pseudo_quantize_tensor(m.weight.data, **q_config)
            pseudo_quantize_tensor(m.weight.data, inplace=True, **q_config)
            m.cpu()
            clear_memory()


@torch.no_grad()
def real_quantize_model_weight(model, q_config, init_only=False):

    q_bit = q_config["q_bit"]

    layers = get_layers(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = AWQLinear.from_linear(
                    module, q_bit, q_config["q_group_size"], True
                )
                q_linear.to(get_device(layer))
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, get_scale_zp=True, **q_config
                )
                q_linear = AWQLinear.from_linear(
                    module, q_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(get_device(layer))
                set_op_by_name(layer, name, q_linear)
                clear_memory()

    clear_memory()
