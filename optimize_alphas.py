import os
import sys
import json
import random
import copy
import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bayes_opt import BayesianOptimization
import argparse
import datetime

# --- START: Logging & System Helpers ---

PROGRESS_WRITER = None

def get_progress_writer():
    """
    Gets a file object for progress updates that bypasses stdout/stderr redirection.
    This writes directly to the controlling terminal.
    """
    global PROGRESS_WRITER
    if PROGRESS_WRITER:
        return PROGRESS_WRITER
    
    try:
        PROGRESS_WRITER = open("/dev/tty", "w")
    except Exception:
        PROGRESS_WRITER = sys.stderr
    return PROGRESS_WRITER

class DualOutput:
    """
    Redirects print statements to both the console (stdout)
    and a log file.
    """
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

# --- 1. COPIED HELPER FUNCTIONS ---
print("Loading helper functions...")

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

def layer_fusion(model, layer1_idx, layer2_idx, ratio_i, weight_types):
    """
    Fuses two specified layers of the model. (CORRECTED VERSION)
    """
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
            # This is the corrected in-place update
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
    writer = get_progress_writer()
    cors = []
    
    model.eval()
    
    num_questions = test_df.shape[0]
    print(f"Evaluating {subject} ({num_questions} questions): ", end="", file=writer, flush=True)

    for i in range(num_questions):
        if (i > 0) and (i % 10 == 0):
            print(".", end="", file=writer, flush=True)
            
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, ntrain)
        prompt = train_prompt + prompt_end
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, -1, :]
        
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().float().cpu().numpy()
        pred = choices[np.argmax(probs[:, [tokenizer(c).input_ids[-1] for c in choices]])]
        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        cor = pred == label
        cors.append(cor)
    
    print(" Done.", file=writer, flush=True)

    acc = np.mean(cors)
    return acc, 0.0

# --- 2. GLOBAL VARIABLES (to be set by args) ---
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3-8B" 
DATA_DIR = "./data"
STARTING_ALPHAS_FILE = "./fusion_alphas.json" 
NUM_LAYERS_TO_FUSE = 13

GLOBAL_BASE_MODEL = None
GLOBAL_TOKENIZER = None
ANATOMY_DEV_DF = None
ANATOMY_TEST_DF = None

WEIGHT_TYPES = [
    "mlp.down_proj.weight", "mlp.up_proj.weight", "mlp.gate_proj.weight",
    "self_attn.k_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_proj.weight", "self_attn.v_proj.weight",
]

def load_globals():
    """Loads the base model, tokenizer, and data ONCE."""
    global GLOBAL_BASE_MODEL, GLOBAL_TOKENIZER, ANATOMY_DEV_DF, ANATOMY_TEST_DF
    
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    GLOBAL_BASE_MODEL = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
    )
    GLOBAL_BASE_MODEL.config.use_cache = False
    
    print(f"Loading tokenizer...")
    GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, use_fast=True, trust_remote_code=True, padding_side="left"
    )
    if GLOBAL_TOKENIZER.pad_token is None:
        GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
        GLOBAL_TOKENIZER.pad_token_id = GLOBAL_TOKENIZER.eos_token_id

    print("Loading 'anatomy' dev and test data...")
    subject = "anatomy"
    ANATOMY_DEV_DF = pd.read_csv(
        os.path.join(DATA_DIR, "dev", subject + "_dev.csv"), header=None
    )[:5]
    ANATOMY_TEST_DF = pd.read_csv(
        os.path.join(DATA_DIR, "test", subject + "_test.csv"), header=None
    )
    print("Globals loaded successfully.")

# --- 3. THE "BLACK BOX" OBJECTIVE FUNCTION ---
def objective_function(alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9, alpha_10, alpha_11, alpha_12):
    writer = get_progress_writer()
    alphas = [
        alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, 
        alpha_7, alpha_8, alpha_9, alpha_10, alpha_11, alpha_12
    ]
    
    print(f"\n--- Starting new evaluation run ---", file=writer, flush=True)
    print(f"Testing alphas: {[f'{a:.4f}' for a in alphas]}", file=writer, flush=True)
    
    try:
        model = copy.deepcopy(GLOBAL_BASE_MODEL)
        model = model.to("cuda") 
        num_layers = model.config.num_hidden_layers

        for i in range(NUM_LAYERS_TO_FUSE):
            layer1_idx = num_layers - 2
            layer2_idx = num_layers - 1
            ratio_i = alphas[i] 
            model = layer_fusion(model, layer1_idx, layer2_idx, ratio_i, WEIGHT_TYPES)
            num_layers -= 1

        model.config.num_hidden_layers = num_layers

        acc, _ = eval_subject(
            "anatomy", 
            model, 
            GLOBAL_TOKENIZER, 
            ANATOMY_DEV_DF, 
            ANATOMY_TEST_DF, 
            ntrain=5
        )
        
        print(f"Run complete. Accuracy: {acc:.4f}", file=writer, flush=True)

        del model
        clear_memory()
        
        return acc

    except Exception as e:
        print(f"Error during evaluation: {e}", file=writer, flush=True)
        clear_memory()
        return 0.0

# --- 4. MAIN OPTIMIZATION SCRIPT ---
if __name__ == "__main__":
    
    # --- 1. Parse Arguments ---
    parser = argparse.ArgumentParser(description="Find optimal fusion alphas using Bayesian Optimization.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Path to the base pre-trained model.")
    parser.add_argument("--alphas_file", type=str, default="./fusion_alphas.json", help="Path to the '.json' file containing *starting* ratios.")
    parser.add_argument("--num_layer", type=int, default=13, help="Number of layers to fuse.")
    parser.add_argument("--output_dir", type=str, default="./optimization_run", help="Directory to save the logs and resulting alphas.json.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the MMLU data.")
    args = parser.parse_args()

    # --- 2. Setup Output Directory & Logging ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_FILENAME = f"optimization_run_{timestamp}.log"
    OUTPUT_PATH = os.path.join(args.output_dir, OUTPUT_FILENAME)
    
    original_stdout = sys.stdout
    sys.stdout = DualOutput(original_stdout, OUTPUT_PATH)
    
    print(f"Bayesian Optimization Script Started")
    print(f"Runtime output will be saved to: {OUTPUT_PATH}")
    print(f"Arguments: {args}")
    
    # --- 3. Set Globals from Args ---
    BASE_MODEL_PATH = args.model_path
    DATA_DIR = args.data_dir
    STARTING_ALPHAS_FILE = args.alphas_file
    NUM_LAYERS_TO_FUSE = args.num_layer

    set_seed(1)
    
    load_globals() 

    # --- 4. Setup Optimizer ---
    pbounds = {}
    for i in range(NUM_LAYERS_TO_FUSE):
        pbounds[f'alpha_{i}'] = (0.3, 0.7) 

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=1,
        verbose=2 
    )

    print(f"\nProbing with known alphas from {STARTING_ALPHAS_FILE}...")
    try:
        with open(STARTING_ALPHAS_FILE, 'r') as f:
            starting_alphas_data = json.load(f)
        
        starting_params = {}
        for i in range(NUM_LAYERS_TO_FUSE):
            starting_params[f'alpha_{i}'] = starting_alphas_data[i]['ratio_i_alpha']
            
        optimizer.probe(
            params=starting_params,
            lazy=True, 
        )
        print("Successfully added starting alphas as first point to test.")
    except Exception as e:
        print(f"Warning: Could not load starting alphas. Proceeding with random start. Error: {e}")

    # --- 5. Run Optimizer ---
    print("\n" + "="*60)
    print("STARTING BAYESIAN OPTIMIZATION")
    print("This will run the 'fuse-then-evaluate' loop multiple times.")
    print("Each 'step' (e.g., 1/25) will take a few minutes.")
    print("="*60)
    
    optimizer.maximize(
        init_points=5, 
        n_iter=25      
    )

    # --- 6. Save Final Results ---
    print("\n" + "="*60)
    print("OPTIMIZATION FINISHED")
    print("="*60)
    print("Best accuracy found:")
    print(optimizer.max)
    
    best_params = optimizer.max['params']
    best_alphas_list = []
    base_num_layers = GLOBAL_BASE_MODEL.config.num_hidden_layers
    for i in range(NUM_LAYERS_TO_FUSE):
        best_alphas_list.append({
            "step": i,
            "layer_to_merge_into": (base_num_layers - 2) - i,
            "layer_to_remove": (base_num_layers - 1) - i,
            "ratio_i_alpha": best_params[f'alpha_{i}']
        })
    
    # Save the final JSON inside the new output directory
    output_file = os.path.join(args.output_dir, "optimized_alphas.json")
    with open(output_file, 'w') as f:
        json.dump(best_alphas_list, f, indent=2)
        
    print(f"\nBest alpha values saved to {output_file}")
    print("You can now use this file with your 'custom_alpha_load_eval.py' script.")
    
    # --- 7. Cleanup ---
    if PROGRESS_WRITER and PROGRESS_WRITER != sys.stderr:
        PROGRESS_WRITER.close()
        
    sys.stdout.close()
    sys.stdout = original_stdout
    
    print(f"\nOptimization finished. All logs and results saved in: {args.output_dir}")