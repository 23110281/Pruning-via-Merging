import os
import torch
from transformers import (
    AutoModelForCausalLM,  
    AutoTokenizer,        
)
import argparse  
import time      
import numpy as np  
import json          
from tqdm import tqdm 
import random         
import pandas as pd    
from sklearn.feature_selection import mutual_info_regression  
from sklearn.neighbors import NearestNeighbors              
import pickle          
import logging         
import gc              

# Define the possible choices for multiple-choice questions
choices = ["A", "B", "C", "D"]

def format_subject(subject):
    """
    Formats the subject string by replacing underscores with spaces.

    Args:
        subject (str): The subject string with underscores.

    Returns:
        str: The formatted subject string with spaces.
    """
    # Split the subject by underscores
    l = subject.split("_")
    s = ""
    # Concatenate each part with a space
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    """
    Formats a single example from the DataFrame into a string prompt.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        idx (int): The index of the row to format.
        include_answer (bool): Whether to include the correct answer.

    Returns:
        str: The formatted example string.
    """
    # Extract the question prompt from the first column
    prompt = df.iloc[idx, 0]
    # Determine the number of choices based on DataFrame columns
    k = df.shape[1] - 2
    # Append each choice to the prompt
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    # Add the "Answer:" prompt
    prompt += "\nAnswer:"
    # Optionally include the correct answer
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    """
    Generates a prompt containing multiple training examples for the given subject.

    Args:
        train_df (pd.DataFrame): The DataFrame containing training data.
        subject (str): The subject name.
        k (int, optional): Number of training examples to include. Defaults to -1 (all).

    Returns:
        str: The generated prompt string.
    """
    # Start the prompt with a description of the task
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    # If k is not specified, use all training examples
    if k == -1:
        k = train_df.shape[0]
    # Append each training example to the prompt
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    """
    Evaluates the model on the test dataset for a specific subject.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        subject (str): The subject name.
        model (torch.nn.Module): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        dev_df (pd.DataFrame): Development set DataFrame.
        test_df (pd.DataFrame): Test set DataFrame.

    Returns:
        tuple: (list of correctness for each example, accuracy, perplexity)
    """
    cors = []          # List to store correctness of each prediction
    all_probs = []     # List to store probabilities (unused in current code)
    total_loss = 0     # Accumulator for total loss to compute perplexity

    # Iterate over each test example with a progress bar
    for i in tqdm(range(test_df.shape[0]), desc=f"Evaluating {subject}"):
        k = args.ntrain  # Number of training examples to include in the prompt
        # Format the current test example without the answer
        prompt_end = format_example(test_df, i, include_answer=False)
        # Generate the training prompt with k examples
        train_prompt = gen_prompt(dev_df, subject, k)
        # Combine training prompt and test example prompt
        prompt = train_prompt + prompt_end
        # Tokenize the combined prompt and move to GPU
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        # Clone input_ids for labels
        labels = input_ids.clone()
        # Mask the training part of the prompt in labels by setting them to -100
        labels[:, :-len(tokenizer(prompt_end).input_ids)] = -100

        # Forward pass through the model to get outputs
        outputs = model(input_ids=input_ids, labels=labels)
        # Extract logits for the last token
        logits = outputs.logits[:, -1, :]
        # Extract loss for the current example
        loss = outputs.loss

        # Accumulate the loss
        total_loss += loss.item()

        # Compute probabilities using softmax on logits
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().float().cpu().numpy()
        # Determine the predicted choice by selecting the choice with the highest probability
        pred = choices[np.argmax(probs[:, [tokenizer(c).input_ids[-1] for c in choices]])]
        # Extract the true label from the test DataFrame
        label = test_df.iloc[i, test_df.shape[1] - 1]

        # Check if the prediction is correct
        cor = pred == label
        cors.append(cor)

    # Calculate average accuracy
    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    # Calculate the average loss and then the perplexity
    avg_loss = total_loss / len(test_df)
    ppl = np.exp(avg_loss)
    print("Perplexity {:.3f} - {}".format(ppl, subject))

    return cors, acc, ppl

def set_seed(seed: int = 1):
    """
    Sets the random seed for reproducibility across various libraries and environments.

    Args:
        seed (int, optional): The seed value to set. Defaults to 1.
    """
    random.seed(seed)  # Set seed for Python's random module
    np.random.seed(seed)  # Set seed for NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set seed for Python hash-based operations
    torch.manual_seed(seed)  # Set seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False     # Disable cuDNN benchmark for consistency

def adaptive_chunk_size(total_size, preferred_size=100):
    """
    Determines the optimal chunk size for processing to maximize efficiency.

    Args:
        total_size (int): The total number of elements to process.
        preferred_size (int, optional): The preferred chunk size. Defaults to 100.

    Returns:
        int: The adaptive chunk size.
    """
    # Iterate from preferred_size down to 1 to find the largest divisor of total_size
    for size in range(preferred_size, 0, -1):
        if total_size % size == 0:
            return size
    return 1  # Fallback to 1 if no divisor is found

def L2_distance_chunked(a, b, df, total_size):
    """
    Generates L2 distance chunks between two arrays in an adaptive chunked manner.

    Args:
        a (np.ndarray): First array of shape (n_samples_a, n_features).
        b (np.ndarray): Second array of shape (n_samples_b, n_features).
        df (int): Flag to determine if diagonal should be zeroed.
        total_size (int): Total number of samples.

    Yields:
        np.ndarray: A chunk of L2 distances.
    """
    # Determine the chunk size adaptively
    chunk_size = adaptive_chunk_size(total_size)
    # Reshape a and b if they have more than 2 dimensions
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-1])

    # Ensure a and b have the same number of features
    assert a.shape[1] == b.shape[1], "Incompatible shapes"

    # Iterate over chunks of a
    for i in range(0, a.shape[0], chunk_size):
        # Compute squared norms for the current chunk of a
        aa = np.sum(a[i : i + chunk_size] ** 2, axis=1, keepdims=True)
        # Iterate over chunks of b
        for j in range(0, b.shape[0], chunk_size):
            # Compute squared norms for the current chunk of b
            bb = np.sum(b[j : j + chunk_size] ** 2, axis=1, keepdims=True).T
            # Compute the dot product between chunks of a and b
            ab = a[i : i + chunk_size] @ b[j : j + chunk_size].T
            # Compute the L2 distance chunk
            d_chunk = np.sqrt(np.abs(aa + bb - 2 * ab))

            # If df flag is set to 1 and processing diagonal chunks, set diagonal to 0
            if df == 1:
                if i == j:
                    np.fill_diagonal(d_chunk, 0)  # Set diagonal to 0 if needed

            # Yield the computed distance chunk
            yield d_chunk

def diffusionKernel(X, sigmaK, alpha, d, total_size):
    """
    Computes the diffusion kernel embedding for the dataset X.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        sigmaK (float): Kernel scale parameter.
        alpha (float): Scaling factor for normalization.
        d (int): Target dimensionality for embedding.
        total_size (int): Total number of samples.

    Returns:
        np.ndarray: Embedded data of shape (n_samples, d).
    """
    # Determine the optimal chunk size for processing
    chunk_size = adaptive_chunk_size(total_size)
    print("Starting diffusion kernel computation...")
    kernel_start_time = time.time()

    n = X.shape[0]  # Number of samples
    # Initialize the kernel matrix with zeros
    K = np.zeros((n, n), dtype=np.float32)

    # Iterate over chunks of X to compute the kernel matrix
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            j_end = min(j + chunk_size, n)
            # Compute the L2 distance chunk between X[i:i_end] and X[j:j_end]
            D_chunk = next(L2_distance_chunked(X[i:i_end], X[j:j_end], df=1, total_size=n))
            # Compute the kernel chunk using the diffusion kernel formula
            K_chunk = np.exp(-((D_chunk / sigmaK) ** 0.5))
            # Assign the computed chunk to the appropriate position in K
            K[i:i_end, j:j_end] = K_chunk[: i_end - i, : j_end - j]

    # Calculate the sum of the kernel matrix along columns
    p = np.sum(K, axis=0)
    # Normalize the kernel matrix
    K1 = K / (p * p.reshape(-1, 1)) ** alpha
    # Compute the normalization factor
    v = np.sqrt(np.sum(K1, axis=0))
    # Normalize the kernel matrix further
    A = K1 / np.outer(v, v)

    # Compute the condition number of the matrix A for numerical stability
    cond_num = np.linalg.cond(A)
    print(f"Condition number: {cond_num}")

    # If the condition number is infinite, apply regularization to stabilize
    if np.isinf(cond_num):
        print("Infinite condition number detected. Applying regularization...")
        regularization = 1e-6
        max_iterations = 10
        iteration = 0
        while np.isinf(cond_num) and iteration < max_iterations:
            # Add a small value to the diagonal for regularization
            A += np.eye(A.shape[0]) * regularization
            cond_num = np.linalg.cond(A)
            regularization *= 10  # Increase regularization factor exponentially
            iteration += 1
        print(f"Regularization applied. New condition number: {cond_num}")

    # Replace any NaNs in A with zero
    A = np.nan_to_num(A)

    # Handle very small values by setting them to a minimum threshold
    zero_mask = np.abs(A) < 1e-12
    A[zero_mask] = 1e-12

    # Perform Singular Value Decomposition (SVD) on the matrix A
    U, S, V = np.linalg.svd(A, full_matrices=False)
    # Retain only the top (d + 1) singular vectors
    U = U[:, :d + 1]
    # Avoid division by zero by replacing zeros in the first column
    U[:, 0] = np.where(U[:, 0] == 0, 1e-8, U[:, 0])
    # Normalize U by the first column
    U = U / U[:, 0].reshape(-1, 1)

    # Extract the embedded coordinates excluding the first column
    Y = U[:, 1 : d + 1]

    kernel_end_time = time.time()
    print(f"Diffusion kernel computation completed in {kernel_end_time - kernel_start_time:.2f} seconds.")
    return Y

def extract_layer_params(model, layer_idx, input_ids):
    """
    Extracts the activations from a specific layer of the model given input tokens.

    Args:
        model (torch.nn.Module): The language model.
        layer_idx (int): The index of the layer to extract.
        input_ids (torch.Tensor): Tokenized input IDs.

    Returns:
        np.ndarray: Activations from the specified layer, adjusted to a maximum length of 512.
    """
    # Perform a forward pass with no gradient computation to get hidden states
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # List of hidden states from each layer
        # Extract activations from the specified layer and move to CPU
        activations = hidden_states[layer_idx].detach().float().cpu().numpy()

    # Define the maximum sequence length
    max_length = 512
    # If the sequence length is shorter than max_length, pad with zeros
    if activations.shape[1] < max_length:
        padding = max_length - activations.shape[1]
        activations = np.pad(activations, ((0, 0), (0, padding), (0, 0)), "constant")
    # If the sequence length is longer than max_length, truncate
    elif activations.shape[1] > max_length:
        activations = activations[:, :max_length, :]

    return activations

def load_embeddings(directory_path):
    """
    Loads and preprocesses layer embeddings from pickle files in the specified directory.

    Args:
        directory_path (str): Path to the directory containing embedding files.

    Returns:
        list: A list of NumPy arrays containing embeddings for each layer.
    """
    embeddings = []  # List to store embeddings from each file
    # Sort filenames based on the numerical value after the first underscore
    filenames = sorted(
        os.listdir(directory_path), key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    # Iterate over each file in the sorted list
    for filename in filenames:
        if filename.endswith(".pkl"):  # Process only pickle files
            with open(os.path.join(directory_path, filename), "rb") as f:
                embedding = pickle.load(f)
                # Replace NaNs and infinite values with zeros
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

                # Apply rank normalization to the embeddings
                embedding = (
                    np.argsort(np.argsort(embedding, axis=0), axis=0)
                    / embedding.shape[0]
                )

                # Append the preprocessed embedding to the list
                embeddings.append(embedding)
    return embeddings

def entropy_estimator_knn(x, k=1):
    """
    Estimates the entropy of the dataset x using a k-nearest neighbors approach.

    Args:
        x (np.ndarray): Input data of shape (n_samples, n_features).
        k (int, optional): Number of neighbors to consider. Defaults to 1.

    Returns:
        float: Estimated entropy.
    """
    n, d = x.shape  # Number of samples and dimensions
    # Initialize the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x)
    # Compute the distances to the nearest neighbors
    distances, _ = nbrs.kneighbors(x)
    # Take the distance to the k-th neighbor (excluding the point itself)
    distances = distances[:, -1]
    # Compute the entropy estimate using the KNN formula
    return -np.mean(np.log(k / (n * distances**d)))

def compute_similarity_matrix_npib_global(embeddings, n_neighbors=5, k_entropy=50):
    """
    Computes a similarity matrix between different layers based on normalized pointwise information bottleneck (NPIB).

    Args:
        embeddings (list): List of NumPy arrays containing embeddings for each layer.
        n_neighbors (int, optional): Number of neighbors for mutual information computation. Defaults to 5.
        k_entropy (int, optional): Number of neighbors for entropy estimation. Defaults to 50.

    Returns:
        np.ndarray: The computed similarity matrix of shape (num_layers, num_layers).
    """
    num_layers = len(embeddings)  # Number of layers
    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((num_layers, num_layers))

    # Iterate over each pair of layers
    for i in range(num_layers):
        for j in range(i, num_layers):
            emb_i = embeddings[i]  # Embeddings for layer i
            emb_j = embeddings[j]  # Embeddings for layer j

            # Ensure both embeddings have the same number of samples by taking the minimum
            min_samples = min(emb_i.shape[0], emb_j.shape[0])
            emb_i = emb_i[:min_samples, :]
            emb_j = emb_j[:min_samples, :]

            # List to store mutual information scores for each dimension
            mi_scores = []
            # Compute mutual information between each dimension of emb_j and the entire emb_i
            for dim in range(emb_j.shape[1]):
                mi_score = mutual_info_regression(
                    emb_i,
                    emb_j[:, dim],
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                )
                # Take the mean mutual information score for the current dimension
                mi_scores.append(np.mean(mi_score))

            # Compute the average mutual information across all dimensions
            mutual_info = np.mean(mi_scores)
            # Estimate the entropy for both embeddings
            entropy_i = entropy_estimator_knn(emb_i, k=k_entropy)
            entropy_j = entropy_estimator_knn(emb_j, k=k_entropy)
            # Compute the normalized pointwise information bottleneck (NPIB)
            npib = mutual_info / np.sqrt(entropy_i * entropy_j)

            # Assign the computed similarity to the matrix (symmetrically)
            similarity_matrix[i, j] = npib
            similarity_matrix[j, i] = npib

    return similarity_matrix

def compute_fusion_ratios(similarity_matrix, sorted_pairs, beta=1.0):
    """
    Computes fusion ratios based on the similarity matrix and sorted layer pairs.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix between layers.
        sorted_pairs (list of tuples): List of layer index pairs to fuse.
        beta (float, optional): Scaling factor for the fusion ratio. Defaults to 1.0.

    Returns:
        list of tuples: List containing (ratio_i, ratio_j) for each pair.
    """
    fusion_ratios = []  # List to store fusion ratios for each pair
    # Iterate over each sorted pair of layers
    for i, j in sorted_pairs:
        # Compute the mean similarity for each layer across all other layers
        similarity_i = np.mean(similarity_matrix[i, :])
        similarity_j = np.mean(similarity_matrix[j, :])
        # Compute the total similarity for normalization
        total_similarity = similarity_i + similarity_j

        # Calculate the ratio for each layer based on their similarity
        ratio_i = similarity_i / total_similarity
        ratio_j = similarity_j / total_similarity

        # Apply a sigmoid-like adjustment to the ratios using beta
        adjusted_ratio_i = np.exp(beta * ratio_i) / (1 + np.exp(beta * ratio_i))
        adjusted_ratio_j = 1 - adjusted_ratio_i

        # Append the adjusted ratios as a tuple
        fusion_ratios.append((adjusted_ratio_i, adjusted_ratio_j))

    return fusion_ratios    

def evaluate(model, tokenizer, args):
    """
    Evaluates the model across all specified subjects and computes accuracy and perplexity.

    Args:
        model (torch.nn.Module): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: Dictionaries containing accuracy and perplexity for each subject.
    """
    model.eval()  # Set the model to evaluation mode

    # Identify all subjects by listing test files and extracting subject names
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test")) 
            if "_test.csv" in f
        ]
    )
    all_accs = {}  # Dictionary to store accuracy for each subject
    all_ppls = {}  # Dictionary to store perplexity for each subject

    # Iterate over each subject
    for subject in subjects:
        # Load the development set for the current subject and take the first k examples
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None  
        )[: args.ntrain]
        # Load the test set for the current subject
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        
        # Evaluate the model on the current subject's test set
        _, acc, ppl = eval(args, subject, model, tokenizer, dev_df, test_df)
        
        # Store the accuracy and perplexity
        all_accs[subject] = acc
        all_ppls[subject] = ppl
        
    model.train()  # Set the model back to training mode
    return all_accs, all_ppls

def clear_memory():
    """
    Clears Python and CUDA memory to free up resources.
    """
    gc.collect()  # Trigger garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Empty CUDA cache if available      

def layer_fusion(model, layer1_idx, layer2_idx, ratio_i, weight_types):
    """
    Fuses two specified layers of the model by blending their weights based on given ratios.

    Args:
        model (torch.nn.Module): The language model.
        layer1_idx (int): Index of the first layer to fuse.
        layer2_idx (int): Index of the second layer to fuse.
        ratio_i (float): Fusion ratio for the first layer.
        weight_types (list): List of weight attribute names to fuse.

    Returns:
        torch.nn.Module: The model after layer fusion.
    """
    print(f"Starting fusion of layers {layer1_idx} and {layer2_idx} with ratio {ratio_i}")

    # Retrieve parameters from the first layer based on weight types
    layer1_params = {
        name: param
        for name, param in model.named_parameters()
        if f"model.layers.{layer1_idx}." in name
    }
    # Retrieve parameters from the second layer based on weight types
    layer2_params = {
        name: param
        for name, param in model.named_parameters()
        if f"model.layers.{layer2_idx}." in name
    }

    # Display parameters of the first layer before fusion
    print(f"Layer {layer1_idx} parameters before fusion:")
    for name in layer1_params:
        print(f"{name}: {layer1_params[name].shape}")

    # Display parameters of the second layer before fusion
    print(f"Layer {layer2_idx} parameters before fusion:")
    for name in layer2_params:
        print(f"{name}: {layer2_params[name].shape}")

    # Fuse each specified weight type
    for weight_type in weight_types:
        # Get weights from both layers
        w1 = layer1_params.get(f"model.layers.{layer1_idx}.{weight_type}")
        w2 = layer2_params.get(f"model.layers.{layer2_idx}.{weight_type}")
        if w1 is not None and w2 is not None:
            ratio_j = 1 - ratio_i  # Complementary ratio for the second layer
            # Compute the fused weights as a weighted sum of both layers' weights
            w_fused = ratio_i * w1.detach().float().cpu().numpy() + ratio_j * w2.detach().float().cpu().numpy()
            # Convert the fused weights back to a PyTorch tensor and move to the appropriate device
            w_fused_tensor = torch.tensor(w_fused).to(w1.device)
            # Update the model's state dictionary with the fused weights
            model.state_dict()[f"model.layers.{layer1_idx}.{weight_type}"] = w_fused_tensor.view_as(w1).to(w1.dtype)

    # Display parameters of the first layer after fusion
    print(f"Layer {layer1_idx} parameters after fusion:")
    for name in layer1_params:
        print(f"{name}: {layer1_params[name].shape}")

    # Remove the second layer from the model's layer list
    model.model.layers = torch.nn.ModuleList(
        [layer for k, layer in enumerate(model.model.layers) if k != layer2_idx]
    )

    print(f"Model layers after removal of layer {layer2_idx}")
    return model

def main():
    """
    The main function that orchestrates the entire process: parsing arguments, loading the model,
    processing data, computing embeddings and similarities, fusing layers, and saving the modified model.
    """
    parser = argparse.ArgumentParser()
    # Define command-line arguments with descriptions and default values
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="Number of training examples to include in prompts")
    parser.add_argument("--ngpu", "-g", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--model_path", type=str, default="./data/yangzhao/point/baichuan/Meta-Llama-3-70B", help="Path to the pre-trained model")
    parser.add_argument("--num_tasks", "-n", type=int, default=57, help="Number of MMLU tasks to process (default: 57)")
    parser.add_argument("--num_samples", "-m", type=int, default=1, help="Number of samples per task (default: 1)")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Directory containing the data")
    parser.add_argument("--num_layer", "-i", type=int, default=1, help="Number of layers to fuse (default: 1)")
    args = parser.parse_args()

    # Extract the model name from the provided model path
    model_name = args.model_path.split("/")[-1]
    # Define the base directory for storing fused model information
    base_dir = f"./data/yangzhao/point/EMNLP2024/layer_fus/al/{model_name}/fused_{args.num_layer}_layers"

    # Define directories for embeddings, fusion info, and merged weights
    iteration_dir = os.path.join(base_dir, f"iteration")
    embeddings_dir = os.path.join(iteration_dir, "embeddings")
    fusion_info_dir = os.path.join(iteration_dir, "fusion_info")
    merged_weights_dir = os.path.join(iteration_dir, "merged_weights")

    # Create the necessary directories if they don't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(fusion_info_dir, exist_ok=True)
    os.makedirs(merged_weights_dir, exist_ok=True)

    # Configure logging to write logs to a file within fusion_info_dir
    logging.basicConfig(filename=os.path.join(fusion_info_dir, 'experiment.log'), level=logging.INFO)
    # Set random seeds for reproducibility
    set_seed(1)

    # Initialize the tokenizer from the pre-trained model
    global tokenizer  # Declare as global to use in other functions if needed
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,             # Use the fast tokenizer implementation
        trust_remote_code=True,    # Trust remote code (required for some models)
        add_bos_token=False,       # Do not add beginning-of-sequence token
        add_eos_token=False,       # Do not add end-of-sequence token
        padding_side="left"        # Pad sequences on the left side
    )

    # Load the pre-trained causal language model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,    # Trust remote code (required for some models)
        device_map="auto",         # Automatically map layers to available devices
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,  # Use bfloat16 if supported
    )

    print(f"Initial model configuration: {model.config}")  # Display the model's configuration

    # Define the types of weights to be fused between layers
    weight_types = [
        "mlp.down_proj.weight",
        "mlp.up_proj.weight", 
        "mlp.gate_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    ]

    # Display metadata about the model
    print("Model metadata:")
    print(f"Number of layers: {len(model.model.layers)}")
    print(f"Config num_hidden_layers: {model.config.num_hidden_layers}")

    # Identify all subjects by listing test files and extracting subject names
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    num_layers = model.config.num_hidden_layers  # Total number of hidden layers in the model
    # Initialize a dictionary to store activations for each layer
    all_layers_activations = {i: [] for i in range(num_layers)}

    # Iterate over each subject up to the specified number of tasks
    for subject in subjects[:args.num_tasks]:
        # Load the test set for the current subject
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        # Determine the number of samples to process for the current subject
        num_samples = min(args.num_samples, test_df.shape[0])
        # Randomly select sample indices from the test set
        sample_indices = random.sample(range(test_df.shape[0]), num_samples)

        # Iterate over each selected sample index with a progress bar
        for index in tqdm(sample_indices, desc=f"Processing {subject}"):
            # Format the test example without the answer
            prompt = format_example(test_df, index, include_answer=False)
            # Tokenize the prompt and move to GPU
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

            # Iterate over each layer to extract activations
            for layer_idx in range(num_layers):
                activations = extract_layer_params(model, layer_idx, input_ids)
                # Append the extracted activations to the corresponding layer's list
                all_layers_activations[layer_idx].append(activations)

            # Clear memory after processing each sample to free up resources
            clear_memory()

    # Apply manifold learning (diffusion kernel) to the stacked activations of each layer
    for layer_idx in range(num_layers):
        # Stack all activations for the current layer vertically
        stacked_activations = np.vstack(all_layers_activations[layer_idx])
        # Compute the embedded activations using the diffusion kernel
        embedded_activations = diffusionKernel(stacked_activations, sigmaK=8, alpha=0.5, d=2, total_size=stacked_activations.shape[0])

        # Define the output file path for the embedded activations
        output_file = os.path.join(embeddings_dir, f"layer_{layer_idx}_embedded.pkl")
        # Save the embedded activations to a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(embedded_activations, f)

    # Load all precomputed embeddings from the embeddings directory
    embeddings = load_embeddings(embeddings_dir)

    # Compute the similarity matrix based on the loaded embeddings
    similarity_matrix = compute_similarity_matrix_npib_global(embeddings)

    all_fusion_ratios = []
    # Merge layers iteratively from the last layer towards the first
    for _ in range(args.num_layer):
        if num_layers <= 1:
            break  # Stop if there is only one layer left

        # Define the indices of the two layers to fuse (last two layers)
        layer1_idx = num_layers - 2
        layer2_idx = num_layers - 1

        # Compute fusion ratios for the current pair of layers based on similarity
        fusion_ratios = compute_fusion_ratios(similarity_matrix, [(layer1_idx, layer2_idx)])
        adjusted_ratio_i, adjusted_ratio_j = fusion_ratios[0]

        all_fusion_ratios.append({
            "step": args.num_layer - num_layers + 1, # e.g., 1, 2, 3...
            "layer_to_merge_into": int(layer1_idx),
            "layer_to_remove": int(layer2_idx),
            "ratio_i_alpha": float(adjusted_ratio_i),
            "ratio_j": float(adjusted_ratio_j)
        })

        print(f"Merging Layer {layer1_idx} (Fusion Ratio: {adjusted_ratio_i:.4f}) and Layer {layer2_idx} (Fusion Ratio: {adjusted_ratio_j:.4f})")

        # Perform the actual layer fusion using the computed ratios
        merged_model = layer_fusion(model, layer1_idx, layer2_idx, adjusted_ratio_i, weight_types)
        model = merged_model  # Update the model with the fused layers

        num_layers -= 1  # Decrement the layer count as one layer has been merged

    # Log the completion of layer fusion
    logging.info(f"Completed layer fusion with {args.num_layer} layers.")

    # Update the model's configuration to reflect the new number of hidden layers
    model.config.num_hidden_layers = num_layers
    # Save the model's configuration to the merged_weights directory
    model.config.save_pretrained(merged_weights_dir)

    # Save the fused model's state dictionary
    state_dict = model.state_dict()

    # Display the keys and tensor shapes from the state dictionary for verification
    print("Model state dict keys and tensor shapes after fusion:")
    for key, tensor in state_dict.items():
        print(f"{key}: {tensor.size()}")

    # Additionally, check and display tensor data types in the state dictionary
    print("\nChecking tensor data types in state dict:")
    for key, tensor in state_dict.items():
        print(f"{key}: {tensor.dtype}")

    # Define the save path for the merged model's state dictionary
    save_path = os.path.join(merged_weights_dir, "pytorch_model.bin")
    # Save the state dictionary to the specified path using PyTorch's save function
    torch.save(state_dict, save_path)
    print(f"Model successfully saved to {save_path}.")

    alphas_path = os.path.join(fusion_info_dir, "fusion_alphas.json")
    try:
        with open(alphas_path, "w") as f:
            json.dump(all_fusion_ratios, f, indent=2)
        print(f"\n✅ Successfully calculated and saved {len(all_fusion_ratios)} fusion ratios to:")
        print(f"{alphas_path}")
        logging.info(f"Saved {len(all_fusion_ratios)} fusion ratios to {alphas_path}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save fusion alphas to {alphas_path}. Error: {e}")
        logging.warning(f"Could not save fusion alphas: {e}")

    # Optional: Print example tensor values from the state dictionary for small tensors
    # This helps in verifying the actual data without overwhelming the output
    print("\nExample tensor values from state dict (limited to small tensors for readability):")
    for key, tensor in state_dict.items():
        if tensor.numel() < 10:  # Only print tensors with fewer than 10 elements
            print(f"{key}: {tensor.tolist()}")

if __name__ == "__main__":
    main()
