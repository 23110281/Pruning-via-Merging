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

# --- Helper Functions (Copied from pipeline.py) ---

def set_seed(seed: int = 1):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_memory():
    """
    Clears Python and CUDA memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      

def layer_fusion(model, layer1_idx, layer2_idx, ratio_i, weight_types):
    """
    Fuses two specified layers of the model. (Copied from pipeline.py)
    """
    print(f"Starting fusion of layers {layer1_idx} and {layer2_idx} with ratio {ratio_i}")
    
    # Get parameters for layer 1
    layer1_params = {
        name: param
        for name, param in model.named_parameters()
        if f"model.layers.{layer1_idx}." in name
    }
    # Get parameters for layer 2
    layer2_params = {
        name: param
        for name, param in model.named_parameters()
        if f"model.layers.{layer2_idx}." in name
    }

    # Blend the weights
    for weight_type in weight_types:
        w1 = layer1_params.get(f"model.layers.{layer1_idx}.{weight_type}")
        w2 = layer2_params.get(f"model.layers.{layer2_idx}.{weight_type}")
        if w1 is not None and w2 is not None:
            ratio_j = 1 - ratio_i
            # Perform fusion using numpy for float32 precision, then cast back
            w_fused = ratio_i * w1.detach().float().cpu().numpy() + ratio_j * w2.detach().float().cpu().numpy()
            w_fused_tensor = torch.tensor(w_fused).to(w1.device)
            # Update the model's state dictionary with the new fused tensor
            model.state_dict()[f"model.layers.{layer1_idx}.{weight_type}"] = w_fused_tensor.view_as(w1).to(w1.dtype)

    # Remove the second layer (layer2_idx) from the model
    model.model.layers = torch.nn.ModuleList(
        [layer for k, layer in enumerate(model.model.layers) if k != layer2_idx]
    )
    return model

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base pre-trained model")
    parser.add_argument("--alphas_file", type=str, required=True, help="Path to the 'fusion_alphas.json' file containing the ratios.")
    parser.add_argument("--num_layer", "-i", type=int, required=True, help="Number of layers to fuse (must match the number of steps in the alphas file you wish to use).")
    parser.add_argument("--output_dir", type=str, default="./custom_alpha_output", help="Directory to save the new fused model, logs, and config.")
    
    args = parser.parse_args()

    # --- 1. Setup Output Directories ---
    # This script saves to its own directory structure
    base_dir = os.path.join(args.output_dir, f"fused_{args.num_layer}_layers_manual")
    iteration_dir = os.path.join(base_dir, f"iteration")
    fusion_info_dir = os.path.join(iteration_dir, "fusion_info")
    merged_weights_dir = os.path.join(iteration_dir, "merged_weights")

    os.makedirs(fusion_info_dir, exist_ok=True)
    os.makedirs(merged_weights_dir, exist_ok=True)

    # Configure logging to save to the new output directory
    logging.basicConfig(filename=os.path.join(fusion_info_dir, 'manual_fusion.log'), level=logging.INFO)
    logging.info(f"Starting manual fusion process.")
    logging.info(f"Args: {args}")
    
    print(f"--- Starting Manual Fusion ---")
    print(f"Model will be saved to: {merged_weights_dir}")
    print(f"Loading Alphas from: {args.alphas_file}")

    set_seed(1)

    # --- 2. Load Alphas File ---
    if not os.path.exists(args.alphas_file):
        print(f"❌ Error: Alphas file not found at {args.alphas_file}")
        logging.error(f"Alphas file not found at {args.alphas_file}")
        sys.exit(1)
        
    with open(args.alphas_file, "r") as f:
        all_fusion_ratios = json.load(f)
        
    if len(all_fusion_ratios) < args.num_layer:
        print(f"❌ Error: Requested to fuse {args.num_layer} layers, but alphas file only contains {len(all_fusion_ratios)} steps.")
        logging.error("Alphas file and --num_layer mismatch.")
        sys.exit(1)
        
    # Select the steps we will use
    fusion_steps_to_run = all_fusion_ratios[:args.num_layer]
    print(f"Loaded {len(all_fusion_ratios)} fusion steps. Will use the first {args.num_layer}.")

    # --- 3. Load Base Model ---
    clear_memory()
    print(f"Loading base model from: {args.model_path}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    logging.info(f"Loaded base model from {args.model_path}")
    num_layers = model.config.num_hidden_layers
    print(f"Base model loaded. Initial layers: {num_layers}")

    # (Weight types from pipeline.py)
    weight_types = [
        "mlp.down_proj.weight", "mlp.up_proj.weight", "mlp.gate_proj.weight",
        "self_attn.k_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_proj.weight", "self_attn.v_proj.weight",
    ]

    # --- 4. Run Fusion Loop ---
    for i, fusion_step in enumerate(fusion_steps_to_run):
        if num_layers <= 1:
            print("Stopping fusion, only one layer left.")
            logging.warning("Fusion stopped early, only one layer left.")
            break
            
        # Get indices and ratio from the loaded file
        layer1_idx = fusion_step["layer_to_merge_into"]
        layer2_idx = fusion_step["layer_to_remove"]
        adjusted_ratio_i = fusion_step["ratio_i_alpha"]

        # --- Validation ---
        # Ensure the model state matches the alpha file's expectations
        expected_l1 = num_layers - 2
        expected_l2 = num_layers - 1
        if layer1_idx != expected_l1 or layer2_idx != expected_l2:
            print(f"❌ Error: Index mismatch in step {i}!")
            print(f"  Loaded step wants to fuse: L{layer1_idx} and L{layer2_idx}")
            print(f"  Model currently has {num_layers} layers. Expected to fuse: L{expected_l1} and L{expected_l2}")
            print("  The alphas file may be from a different model or fusion process. Aborting.")
            logging.error("Index mismatch during fusion. Aborting.")
            sys.exit(1)
        # --- End Validation ---

        print(f"\n--- Fusion Step {i+1}/{args.num_layer} ---")
        print(f"Merging L{layer1_idx} (Alpha: {adjusted_ratio_i:.4f}) into L{layer2_idx}")
        logging.info(f"Step {i+1}: Fusing {layer1_idx} (alpha: {adjusted_ratio_i}) and {layer2_idx}")

        # Perform the actual layer fusion
        merged_model = layer_fusion(model, layer1_idx, layer2_idx, adjusted_ratio_i, weight_types)
        model = merged_model
        
        num_layers -= 1  # Decrement the layer count
        print(f"Model now has {num_layers} layers.")

    logging.info(f"Completed layer fusion. Final layer count: {num_layers}")

    # --- 5. Save Fused Model ---
    # (Copied from pipeline.py, saves to the new merged_weights_dir)
    print("\n--- Saving Fused Model ---")
    model.config.num_hidden_layers = num_layers
    model.config.save_pretrained(merged_weights_dir)
    print(f"Model config saved to {merged_weights_dir}")

    state_dict = model.state_dict()
    print("Model state dict keys and tensor shapes after fusion:")
    for key, tensor in state_dict.items():
        print(f"{key}: {tensor.size()}")

    save_path = os.path.join(merged_weights_dir, "pytorch_model.bin")
    torch.save(state_dict, save_path)
    print(f"\n✅ Model successfully saved to {save_path}.")
    logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()