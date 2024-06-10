from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
args = parser.parse_args()

def main():
    model_path = args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(
        config=config, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = model.eval()
    model.to("cuda")
    
if __name__ == "__main__":
    main()