import os
import torch
from transformers import (
    AutoModelForCausalLM,  
    AutoTokenizer,        
)
import argparse  
import numpy as np  
import json          
import random         
import logging         
import gc              
import sys
import datetime
import pandas as pd
from tqdm import tqdm

def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()        

class DualOutput:
    def __init__(self, stdout, log_file_path):
        self.stdout = stdout
        try:
            self.file = open(log_file_path, 'w', buffering=1)
        except Exception as e:
            self.file = None
            print(f"Warning: Could not open log file {log_file_path}. Output will only go to console. Error: {e}")

    def write(self, text):
        self.stdout.write(text)
        if self.file:
            self.file.write(text)

    def flush(self):
        self.stdout.flush()
        if self.file:
            self.file.flush()
            
    def close(self):
        if self.file:
            self.file.close()

def layer_fusion(model, layer1_idx, layer2_idx, ratio_i, weight_types):
    print(f"Fusing L{layer2_idx} into L{layer1_idx} with alpha={ratio_i:.4f}")
    
    layer1_params = {
        name: param
        for name, param in model.named_parameters()
        if f"model.layers.{layer1_idx}." in name
    }
    layer2_params = {
        name: param
        for name, param in model.named_parameters()
        if f"model.layers.{layer2_idx}." in name
    }

    for weight_type in weight_types:
        w1 = layer1_params.get(f"model.layers.{layer1_idx}.{weight_type}")
        w2 = layer2_params.get(f"model.layers.{layer2_idx}.{weight_type}")
        if w1 is not None and w2 is not None:
            ratio_j = 1 - ratio_i
            with torch.no_grad():
                w_fused = ratio_i * w1.detach().float() + ratio_j * w2.detach().float()
                w1.data.copy_(w_fused.to(w1.dtype))

    model.model.layers = torch.nn.ModuleList(
        [layer for k, layer in enumerate(model.model.layers) if k != layer2_idx]
    )
    return model

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad()
def eval_subject(subject, model, tokenizer, dev_df, test_df, ntrain=5):
    cors = []
    total_loss = 0
    
    for i in tqdm(range(test_df.shape[0]), desc=f"Evaluating {subject}", file=sys.stdout):
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, ntrain)
        prompt = train_prompt + prompt_end
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        labels = input_ids.clone()
        labels[:, :-len(tokenizer(prompt_end).input_ids)] = -100
        
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        logits = outputs.logits[:, -1, :]
        loss = outputs.loss
        total_loss += loss.item()
        
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().float().cpu().numpy()
        pred = choices[np.argmax(probs[:, [tokenizer(c).input_ids[-1] for c in choices]])]
        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        cor = pred == label
        cors.append(cor)
    
    acc = np.mean(cors)
    avg_loss = total_loss / len(test_df)
    ppl = np.exp(avg_loss)
    
    return acc, ppl

def main():
    parser = argparse.ArgumentParser(description="Fuse a model from alphas and evaluate it on MMLU.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base pre-trained model (e.g., 'meta-llama/Meta-Llama-3-8B')")
    parser.add_argument("--alphas_file", type=str, required=True, help="Path to the '.json' file containing the fusion ratios.")
    parser.add_argument("--num_layer", "-i", type=int, required=True, help="Number of layers to fuse.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new fused model, logs, and results (e.g., './parthew')")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the MMLU data.")
    
    args = parser.parse_args()

    base_dir = os.path.join(args.output_dir, f"fused_{args.num_layer}_layers_manual")
    iteration_dir = os.path.join(base_dir, f"iteration")
    fusion_info_dir = os.path.join(iteration_dir, "fusion_info")
    merged_weights_dir = os.path.join(iteration_dir, "merged_weights")

    os.makedirs(fusion_info_dir, exist_ok=True)
    os.makedirs(merged_weights_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(fusion_info_dir, 'manual_fusion.log'), level=logging.INFO)
    logging.info(f"Starting manual fusion process.")
    logging.info(f"Args: {args}")
    
    print(f"--- Starting Manual Fusion ---")
    print(f"Output directory: {args.output_dir}")
    print(f"Model will be saved to: {merged_weights_dir}")
    print(f"Loading Alphas from: {args.alphas_file}")

    set_seed(1)

    if not os.path.exists(args.alphas_file):
        print(f"Error: Alphas file not found at {args.alphas_file}")
        logging.error(f"Alphas file not found at {args.alphas_file}")
        sys.exit(1)
        
    with open(args.alphas_file, "r") as f:
        all_fusion_ratios = json.load(f)
        
    if len(all_fusion_ratios) < args.num_layer:
        print(f"Error: Requested to fuse {args.num_layer} layers, but alphas file only contains {len(all_fusion_ratios)} steps.")
        logging.error("Alphas file and --num_layer mismatch.")
        sys.exit(1)
        
    fusion_steps_to_run = all_fusion_ratios[:args.num_layer]
    print(f"Loaded {len(all_fusion_ratios)} fusion steps. Will use the first {args.num_layer}.")

    clear_memory()
    print(f"Loading base model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    fused_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    logging.info(f"Loaded base model from {args.model_path}")
    num_layers = fused_model.config.num_hidden_layers
    print(f"Base model loaded. Initial layers: {num_layers}")

    weight_types = [
        "mlp.down_proj.weight", "mlp.up_proj.weight", "mlp.gate_proj.weight",
        "self_attn.k_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_proj.weight", "self_attn.v_proj.weight",
    ]

    for i, fusion_step in enumerate(fusion_steps_to_run):
        if num_layers <= 1:
            print("Stopping fusion, only one layer left.")
            logging.warning("Fusion stopped early, only one layer left.")
            break
            
        layer1_idx = fusion_step["layer_to_merge_into"]
        layer2_idx = fusion_step["layer_to_remove"]
        adjusted_ratio_i = fusion_step["ratio_i_alpha"]

        expected_l1 = num_layers - 2
        expected_l2 = num_layers - 1
        if layer1_idx != expected_l1 or layer2_idx != expected_l2:
            print(f"Error: Index mismatch in step {i}!")
            print(f"  Loaded step wants to fuse: L{layer1_idx} and L{layer2_idx}")
            print(f"  Model currently has {num_layers} layers. Expected to fuse: L{expected_l1} and L{expected_l2}")
            logging.error("Index mismatch during fusion. Aborting.")
            sys.exit(1)

        print(f"\n--- Fusion Step {i+1}/{args.num_layer} ---")
        logging.info(f"Step {i+1}: Fusing {layer1_idx} (alpha: {adjusted_ratio_i}) and {layer2_idx}")

        merged_model = layer_fusion(fused_model, layer1_idx, layer2_idx, adjusted_ratio_i, weight_types)
        fused_model = merged_model
        
        num_layers -= 1
        print(f"Model now has {num_layers} layers.")

    logging.info(f"Completed layer fusion. Final layer count: {num_layers}")
    fused_model.config.use_cache = False

    print("\n--- Saving Fused Model ---")
    fused_model.config.num_hidden_layers = num_layers
    fused_model.config.save_pretrained(merged_weights_dir)
    print(f"Model config saved to {merged_weights_dir}")

    state_dict = fused_model.state_dict()
    save_path = os.path.join(merged_weights_dir, "pytorch_model.bin")
    torch.save(state_dict, save_path)
    print(f"\nModel successfully saved to {save_path}.")
    logging.info(f"Model saved to {save_path}")

    print("\n--- Starting MMLU Evaluation ---")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_FILENAME = f"mmlu_fused_eval_{timestamp}.log"
    OUTPUT_PATH = os.path.join(fusion_info_dir, OUTPUT_FILENAME)

    original_stdout = sys.stdout
    sys.stdout = DualOutput(original_stdout, OUTPUT_PATH)

    print(f"MMLU Evaluation Script Started")
    print(f"Runtime output will be saved to: {OUTPUT_PATH}")

    print("\n" + "="*60)
    print("EVALUATING FUSED MODEL ON MMLU")
    print("="*60)

    fused_model.eval()

    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(args.data_dir, "test"))
        if "_test.csv" in f
    ])

    all_accs = {}
    all_ppls = {}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[:5]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        
        acc, ppl = eval_subject(subject, fused_model, tokenizer, dev_df, test_df, ntrain=5)
        
        all_accs[subject] = acc
        all_ppls[subject] = ppl
        
        print(f"Average accuracy {acc:.3f} - {subject}")
        print(f"Perplexity {ppl:.3f} - {subject}")

    avg_acc = np.mean(list(all_accs.values()))
    avg_ppl = np.mean(list(all_ppls.values()))

    print("\n" + "="*60)
    print("MMLU EVALUATION RESULTS")
    print("="*60)
    print(f"Average Accuracy:    {avg_acc:.4f}")
    print(f"Average Perplexity: {avg_ppl:.4f}")
    print("="*60)

    results = {
        "average_accuracy": float(avg_acc),
        "average_perplexity": float(avg_ppl),
        "per_subject_accuracy": {k: float(v) for k, v in all_accs.items()},
        "per_subject_perplexity": {k: float(v) for k, v in all_ppls.items()},
    }

    results_path = os.path.join(fusion_info_dir, "mmlu_results.json")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMMLU results saved to: {results_path}")

    sys.stdout.close()
    sys.stdout = original_stdout

    print(f"\nEvaluation finished. All output, including the results, has been saved to: {fusion_info_dir}")

if __name__ == "__main__":
    main()