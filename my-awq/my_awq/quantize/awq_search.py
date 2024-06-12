from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
import torch
from torch import nn
import tqdm
from collections import defaultdict
import functools
import logging

from .quantizer import pseudo_quantize_tensor
from .awq_module_extract import get_layers, get_embed
from .clipper import auto_clip_layer, apply_clip
from .awq_quantize import apply_awq_scale
from .awq_module_extract import get_named_linears
from ..utils.calib_data import get_calib_dataset
from ..utils.utils import (
    clear_memory,
    get_op_name,
    append_str_prefix,
    get_device,
)

logger = logging.getLogger("myawq")


def get_layer0_input(model, embed, layers, model_input):
    layer0_input = None
    layer_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            nonlocal layer0_input
            layer0_input = inp
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    embed.cuda()
    layers[0] = Catcher(layers[0])
    try:
        model(model_input.to(get_device(embed)))  # multi-gpu
    except ValueError:
        pass
    layers[0] = layers[0].module
    embed.cpu()

    return layer0_input, layer_kwargs


@torch.no_grad()
def get_act_scale(x):
    """
    input = [[-1 ,-2, -3], [4, 5, 6]]

    output = [2.5, 3.5, 4.5]
    """
    return x.abs().view(-1, x.shape[-1]).mean(dim=0)


@torch.no_grad()
def auto_scale_layer(
    module,
    layer_kwargs,
    q_config,
    input_feat,
):

    def _search_module_scale(module2inspect, linears2scale: list, x, kwargs={}):
        """
        return scales: [1, ci]
        """
        # w: co, ci
        # x: n, ci
        x = x.to(get_device(module2inspect))
        with torch.no_grad():
            org_out = module2inspect(x, **kwargs)
            if isinstance(org_out, tuple):
                # LlamaSdpaAttention output (attn_output, None, past_key_value)
                org_out = org_out[0]
        x_mean = get_act_scale(x)

        best_error = float("inf")
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_mean.pow(ratio).clamp(min=1e-4)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(get_device(fc)))
                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, **q_config) / (
                    scales.view(1, -1)
                )
            out = module2inspect(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_scales = scales
            module2inspect.load_state_dict(org_sd)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale(prev_layer, linears2scale, inp, module2inspect=None, kwargs={}):
        scales = _search_module_scale(module2inspect, linears2scale, inp, kwargs)
        scales = scales.detach().cpu()
        # prev_layer_name, [layer_name], scales
        return (
            get_op_name(module, prev_layer),
            tuple([get_op_name(module, m) for m in linears2scale]),
            scales,
        )

    scales_list = []
    if isinstance(module, LlamaDecoderLayer):
        """
        (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaSdpaAttention( # softmax(QK^T)V
                (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP( # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
                (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
                (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
                (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
        )
        """

        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_layer=module.input_layernorm,
                linears2scale=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=layer_kwargs,
            )
        )
        # attn out
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_layer=module.self_attn.v_proj,
                    linears2scale=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                    module2inspect=module.self_attn.o_proj,
                )
            )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_layer=module.post_attention_layernorm,
                linears2scale=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_layer=module.mlp.up_proj,
                linears2scale=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                module2inspect=module.mlp.down_proj,
            )
        )
    else:
        raise NotImplementedError(f"Unsupported layer: {type(module)}")

    return scales_list


@torch.no_grad()
def run_awq_search(
    model: LlamaForCausalLM,
    enc,
    q_config,
    n_samples=128,
    seqlen=512,
    # offline
    calib_dataset_path="mit-han-lab/pile-val-backup",
):
    assert isinstance(model, LlamaForCausalLM), "Only LlamaForCausalLM is supported"

    samples = get_calib_dataset(
        calib_dataset_path, enc, n_samples=n_samples, block_size=seqlen
    )

    embed = get_embed(model)
    layers = get_layers(model)

    inps, layer_kwargs = get_layer0_input(model, embed, layers, samples)

    del samples
    clear_memory()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer.cuda()

        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)
            assert len(feat_dict[name]) == 1, "In Llama, there should be only one input"

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )

        inps.to(get_device(layer))

        inps = layer(inps, **layer_kwargs)[0]
        clear_memory()

        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # if it applies, we should also modify the input_feat with scales
        scales_list = auto_scale_layer(
            layer,
            layer_kwargs,
            q_config=q_config,
            input_feat=input_feat,
        )
        apply_awq_scale(layer, scales_list, input_feat_dict=input_feat)
        # apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
        # append prefix to make names global
        awq_results["scale"] += append_str_prefix(
            scales_list, get_op_name(model, layer) + "."
        )

        clear_memory()

        clip_list = auto_clip_layer(
            layer,
            q_config=q_config,
            input_feat=input_feat,
        )
        apply_clip(layer, clip_list)
        # append prefix to make names global
        awq_results["clip"] += append_str_prefix(
            clip_list, get_op_name(model, layer) + "."
        )

        del input_feat

        layer.cpu()
        clear_memory()

    return awq_results
