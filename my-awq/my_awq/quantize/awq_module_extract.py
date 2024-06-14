from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from torch import nn


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
"""


def get_layers(model):
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers  # (layers): ModuleList
    else:
        raise NotImplementedError(f"Unsupported model: {type(model)}")


def get_embed(model):
    if isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens  # (embed_tokens): Embedding(32000, 4096)
    else:
        raise NotImplementedError(f"Unsupported model: {type(model)}")


def get_awqabled_module(module):
    # pre, linears, inspect, with_layer_kwargs
    results = []
    if isinstance(module, LlamaDecoderLayer):
        results.append(
            [
                module.input_layernorm,
                [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                module.self_attn,
                True,
            ]
        )
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            results.append(
                [
                    module.self_attn.v_proj,
                    [module.self_attn.o_proj],
                    module.self_attn.o_proj,
                    False,
                ]
            )

        results.append(
            [
                module.post_attention_layernorm,
                [module.mlp.gate_proj, module.mlp.up_proj],
                module.mlp,
                False,
            ]
        )
        results.append(
            [
                module.mlp.up_proj,
                [module.mlp.down_proj],
                module.mlp.down_proj,
                False,
            ]
        )
        return results
    else:
        raise NotImplementedError(f"Unsupported layer: {type(module)}")
