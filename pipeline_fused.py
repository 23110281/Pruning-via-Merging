import sys
import os
import datetime
# --- START: Output Redirection Setup ---
# 1. Define the output file path based on current time/date
OUTPUT_DIR = "./output_custom_alpha"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILENAME = f"mmlu_fused_eval_{timestamp}.log"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# 2. Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Custom class to tee output to both console and file
class DualOutput:
    def __init__(self, stdout, log_file_path):
        self.stdout = stdout
        try:
            # Open the file in write mode ('w') with buffering set to 1 (line buffering)
            self.file = open(log_file_path, 'w', buffering=1)
        except Exception as e:
            # Fallback to console only if file creation fails
            self.file = None
            print(f"Warning: Could not open log file {log_file_path}. Output will only go to console. Error: {e}")

    def write(self, text):
        # Write to console
        self.stdout.write(text)
        # Write to file if open
        if self.file:
            self.file.write(text)

    def flush(self):
        # Flush both streams
        self.stdout.flush()
        if self.file:
            self.file.flush()
            
    def close(self):
        if self.file:
            self.file.close()

# 4. Redirect sys.stdout
original_stdout = sys.stdout
sys.stdout = DualOutput(original_stdout, OUTPUT_PATH)

print(f"🚀 Output will be saved to: {OUTPUT_PATH}")
# --- END: Output Redirection Setup ---


MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
DATA_DIR = "./data"

# Load the fused model with learned alphas
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Path to the saved fused model
# NUM_LAYERS is not defined in your script, assuming a value for demonstration/file-path creation
# If the path is only used for the save-results part, it may be fine, but for the load path
# we rely on the hardcoded value below, just keep this in mind.
try:
    # Attempt to use NUM_LAYERS if it's defined elsewhere
    _ = NUM_LAYERS
except NameError:
    # Default/Placeholder if not defined
    NUM_LAYERS = 13

# MODEL_DIR = f"./output/Meta-Llama-3-8B/fused_{NUM_LAYERS}_layers/iteration/merged_weights"
MODEL_DIR = f"./custom_alpha_output_1/fused_13_layers_manual/iteration/merged_weights"

print("Loading fused model with learned alphas...")
print(f"Model directory: {MODEL_DIR}")

# Check if model exists
if not os.path.exists(MODEL_DIR):
    print(f"❌ Model directory not found: {MODEL_DIR}")
    print("    Make sure training has completed successfully.")
    # Exit gracefully if model is not found, or the rest of the script will fail
    sys.exit(1)
else:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Load the fused model
    fused_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    
    # Disable caching
    fused_model.config.use_cache = False
    
    print(f"✅ Model loaded successfully!")
    print(f"    Number of layers: {fused_model.config.num_hidden_layers}")
    print(f"    Model dtype: {fused_model.dtype}")













# Evaluate on MMLU
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import json

# MMLU evaluation function
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
    
    for i in tqdm(range(test_df.shape[0]), desc=f"Evaluating {subject}"):
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

# Run evaluation on all subjects
print("\n" + "="*60)
print("EVALUATING FUSED MODEL ON MMLU")
print("="*60)

fused_model.eval()

subjects = sorted([
    f.split("_test.csv")[0]
    for f in os.listdir(os.path.join(DATA_DIR, "test"))
    if "_test.csv" in f
])

all_accs = {}
all_ppls = {}

for subject in subjects:
    dev_df = pd.read_csv(
        os.path.join(DATA_DIR, "dev", subject + "_dev.csv"), header=None
    )[:5]  # Use 5 examples
    test_df = pd.read_csv(
        os.path.join(DATA_DIR, "test", subject + "_test.csv"), header=None
    )
    
    acc, ppl = eval_subject(subject, fused_model, tokenizer, dev_df, test_df, ntrain=5)
    
    all_accs[subject] = acc
    all_ppls[subject] = ppl
    
    print(f"Average accuracy {acc:.3f} - {subject}")
    print(f"Perplexity {ppl:.3f} - {subject}")

avg_acc = np.mean(list(all_accs.values()))
avg_ppl = np.mean(list(all_ppls.values()))

print("\n" + "="*60)
print("MMLU EVALUATION RESULTS (LEARNED ALPHA)")
print("="*60)
print(f"Average Accuracy:    {avg_acc:.4f}")
print(f"Average Perplexity: {avg_ppl:.4f}")
print("="*60)

# Save results
results = {
    "average_accuracy": float(avg_acc),
    "average_perplexity": float(avg_ppl),
    "per_subject_accuracy": {k: float(v) for k, v in all_accs.items()},
    "per_subject_perplexity": {k: float(v) for k, v in all_ppls.items()},
}

results_path = f"./output/Meta-Llama-3-8B/fused_{NUM_LAYERS}_layers/iteration/fusion_info/mmlu_results.json"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {results_path}")

# --- START: Cleanup ---
# 5. Close the file and restore sys.stdout
sys.stdout.close()
sys.stdout = original_stdout
# --- END: Cleanup ---

print(f"\nEvaluation finished. All output, including the results, has been saved to: {OUTPUT_PATH}")