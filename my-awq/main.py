from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from torch import nn
import argparse
import tqdm
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    help="path of the hf model",
    default="/data/models/llama-2-7b",
)

parser.add_argument(
    "--wikitext_path",
    type=str,
    help="path of the wikitext dataset",
    default="/data/datasets/wikitext",
)

args = parser.parse_args()


def main():
    model_path = args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        **{"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    )
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "LlamaDecoderLayer",
        ],
    )
    model = dispatch_model(model, device_map=device_map)

    model.eval()
    args.wikitext_path = args.wikitext_path if args.wikitext_path else "wikitext"
    testenc = load_dataset(args.wikitext_path, "wikitext-2-raw-v1", split="test")
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())


if __name__ == "__main__":
    main()
