import logging

logger = logging.getLogger("myawq")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse

from .utils.eval import eval_wikitext_ppl
from .quantize.awq_search import run_awq_search
from .quantize.awq_quantize import (
    apply_awq_scale,
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from .quantize.clipper import apply_clip
from accelerate import (
    init_empty_weights,
    load_checkpoint_in_model,
    infer_auto_device_map,
)

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
    help="local path of the wikitext dataset, if you can't access huggingface.co",
    default="wikitext",
)
parser.add_argument(
    "--run_eval", action="store_true", help="run evaluation on the model"
)
parser.add_argument(
    "--kernels",
    type=str,
    default="awq",
    help="quantization kernel, awq or custom",
    choices=["awq", "custom"],
)
parser.add_argument("--w_bit", type=int, default=4)
parser.add_argument("--q_group_size", type=int, default=128)
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# no dump_fake
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--calib_dataset_path", type=str, default="mit-han-lab/pile-val-backup"
)
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
# load_awq (+ calculate fake_quant) = load_fake
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)

args = parser.parse_args()


def main():
    q_config = {
        "q_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "zero_point": True,
    }
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False
    enc = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )

    if args.load_quant:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, q_config=q_config, init_only=True, kernel=args.kernels
        )
        model.tie_weights()
        model.load_state_dict(torch.load(args.load_quant), assign=True)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            **{"torch_dtype": torch.float16, "low_cpu_mem_usage": True},
        )
        model.eval()

        if args.run_awq:
            awq_results = run_awq_search(
                model,
                enc,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
                # offline
                calib_dataset_path=args.calib_dataset_path,
            )
            if args.dump_awq:
                torch.save(awq_results, args.dump_awq)
                logger.info(f"AWQ results saved at {args.dump_awq}")
            exit(0)
        else:
            if args.load_awq:  # dump_real or dump_fake
                awq_results = torch.load(args.load_awq, map_location="cpu")
                apply_awq_scale(model, awq_results["scale"], disabletqdm=False)
                apply_clip(model, awq_results["clip"], disabletqdm=False)

                if args.q_backend == "fake":
                    pseudo_quantize_model_weight(model, q_config)
                else:
                    real_quantize_model_weight(model, q_config, kernel=args.kernel)

                if args.dump_quant:
                    assert args.q_backend != "fake", "Cannot dump fake quant model"
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    logger.info(f"Quantized model saved at {args.dump_quant}")
            elif not args.load_quant:  # test original model
                pass

    if args.run_eval:
        eval_wikitext_ppl(model, enc, args.wikitext_path)


if __name__ == "__main__":
    main()
