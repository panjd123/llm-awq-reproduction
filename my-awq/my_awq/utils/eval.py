import torch
from torch import nn
import tqdm
from accelerate import (
    infer_auto_device_map,
    dispatch_model,
)
from datasets import load_dataset
import logging

logger = logging.getLogger("myawq")


@torch.no_grad()
def eval_wikitext_ppl(model, enc, wikitext_path="wikitext"):
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "LlamaDecoderLayer",
        ],
    )
    model = dispatch_model(model, device_map=device_map)
    model.eval()
    testenc = load_dataset(wikitext_path, "wikitext-2-raw-v1", split="test")
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    bar = tqdm.tqdm(range(nsamples), total=nsamples, desc=f"evaluating (ppl: None) ...")
    for i in bar:
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / ((i + 1) * model.seqlen))
        bar.set_description(f"evaluating (ppl: {ppl.item():.3f}) ...")

    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    logger.info(f"ppl: {ppl.item()}")
    return ppl.item()
