import os #path/directory stuff
import pickle

#Deep learning
import torch
from LearnedKernelModel import *

#Math
from sklearn.metrics import f1_score, r2_score
from scipy.signal import savgol_filter
import numpy as np
import copy
import random

device = "cuda" #device to use

def similarity(v1, v2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        v1, v2 (torch.Tensor): The input vectors.

    Returns:
        float: The cosine similarity between v1 and v2.
    """
    norm_v1 = v1 / v1.norm()
    norm_v2 = v2 / v2.norm()

    return (norm_v1 *norm_v2).sum().item()

def compute_param_num(num_conv1, num_conv2, num_conv3):
    """
    Compute the number of parameters in a network given the number of kernels in each layer.

    Args:
        num_conv1, num_conv2, num_conv3 (int): The number of kernels in each convolutional layer.

    Returns:
        int: The total number of parameters in the network.
    """
    params = num_conv1 *192 +num_conv2 * 96 + num_conv3 * 64  # kernel params
    params += num_conv1 + num_conv2 + num_conv3  # biases (1 per kernel)
    params += num_conv1 + num_conv2 + num_conv3  # weights (1 per kernel)

    return params


def get_most_similar_kernels(similarity_flat, coords):
    """
    Get the indices of the most similar kernels based on their similarity scores.

    Args:
        similarity_flat (np.array): The flattened array of similarity scores.
        coords (np.array): The flattened array of kernel index pairs.

    Returns:
        np.array: The indices of the most similar kernels.
    """
    return coords[np.argsort(similarity_flat)]


def compute_similarity(state_dict, conv_i, num_kernels):
    """
    Compute the similarity between convolutional kernels for a given layer.

    Args:
        state_dict (dict): The state dict of the network.
        conv_i (int): Index of the convolutional layer.
        num_kernels (int): Number of kernels in each layer.

    Returns:
        tuple: Two numpy arrays containing the flattened similarity scores and their corresponding coordinates.
    """
    coords = []
    similarity_flat = []

    # Iterate over all pairs of kernels
    for i in range(num_kernels):
        for j in range(i, num_kernels):
            if i != j:
                sim = similarity(state_dict[f'conv{conv_i}.weight'][i], state_dict[f'conv{conv_i}.weight'][j])
                similarity_flat.append(sim)
                coords.append((j, i))

    return np.asarray(similarity_flat), np.asarray(coords)


def prune(state_dict, conv_i, num_kernels, prune_ratio):
    """
    Prune the least important kernels from a kernel group based on cosine similarity and kernel importance.

    Args:
        state_dict (dict): The state dict of the network.
        conv_i (int): Index of the kernel group.
        num_kernels (int): Number of kernels in each group.
        prune_ratio (float): The proportion of kernels to prune.

    Returns:
        dict: The updated state dict after pruning.
    """
    # Compute similarity of kernels
    sim_flat, coords = compute_similarity(state_dict, conv_i, num_kernels)

    # Get the most similar kernels
    most_similar_kernels = get_most_similar_kernels(sim_flat, coords)

    # Prune if the ratio is greater than zero, otherwise do nothing
    if prune_ratio > 0:
        # Iterate over the most similar kernels
        for item in most_similar_kernels[-int(num_kernels * prune_ratio):]:
            # Calculate weights for two kernels under consideration
            item0_weight = state_dict[f'w{conv_i}'][item[0]] * state_dict[f'conv{conv_i}.weight'][item[0]].abs().mean()
            item1_weight = state_dict[f'w{conv_i}'][item[1]] * state_dict[f'conv{conv_i}.weight'][item[1]].abs().mean()

            # Decide which kernel to remove and which to keep
            remove, keep, keep_weight, remove_weight = (
            item[1], item[0], item0_weight, item1_weight) if item0_weight > item1_weight else (
            item[0], item[1], item1_weight, item0_weight)

            # Update state_dict
            state_dict[f'w{conv_i}'][keep] = (keep_weight + remove_weight) / state_dict[f'conv{conv_i}.weight'][
                keep].abs().mean()
            state_dict[f'conv{conv_i}.bias'][keep] += state_dict[f'conv{conv_i}.bias'][remove]

            # this step is actually enough to "prune the kernel", for computation/measurement sake. In reality, we'd want to remove the kernel from the network for a speedup
            state_dict[f'w{conv_i}'][remove] = 0.0
            state_dict[f'conv{conv_i}.bias'][remove] = 0.0

            # this is an extra step used for counting the kernel that we remove in the end
            # no matter what this value is set to, it will not have any effect on the network since the weight is set to zero
            # however, this value *does* need to be non-zero, since PyTorch handles completely zeroed convolutions a bit weirdly
            # and you'll get weird results convolving a purely 0 kernel
            state_dict[f'conv{conv_i}.weight'][remove] = 1e-5

    return state_dict


def prune_network(net, num_kernels, prune_ratio):
    """
    Prune the least important kernels from the model.

    Args:
        net (nn.Module): The model.
        num_kernels (int): Number of kernels in each group.
        prune_ratio (list): The proportion of kernels to prune in each group.

    Returns:
        nn.Module: The pruned network.
    """
    state_dict = net.state_dict()

    # Iterate over all layers and prune
    for conv_i in range(1, 4):
        state_dict = prune(state_dict, conv_i, num_kernels, prune_ratio[conv_i - 1])
    net.load_state_dict(state_dict)
    return net


def count_nonzero_weights(state_dict, num_kernels):
    """
    Count the number of non-zero weights in each kernel group of the model.

    Args:
        state_dict (dict): The state dict of the network.
        num_kernels (int): Number of kernels in each layer.

    Returns:
        list: The number of non-zero weights in each layer.
    """
    zero_weights = [0, 0, 0]

    # Iterate over all layers and kernels
    for i in range(1, 4):
        for j in range(0, num_kernels):
            # Count the zero weights
            if (state_dict[f'conv{i}.weight'][
                    j] == 1e-5).all():  # this is the value we set the weights to in the prune function, it is arbitrary
                zero_weights[i - 1] += 1
    nonzero_weights = [num_kernels - zero_weights[i] for i in range(3)]

    return nonzero_weights

filter_nums = [4, 8, 16, 24, 32, 64, 128, 256, 512] #number of filters to train models for
folds = 10 #number of folds to use for cross validation
save_dir = "models" #directory to save models to
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from KernelLernelTestModel import *

# Initiate lists to store results
pre_prunes = []  # DICE scores before pruning
post_prunes = []  # DICE scores after pruning
reductions = []  # Reduction in parameters
num_kernels = 128  # Number of kernels in each kernel group

# Iterate over models and prune
for j in range(folds):
    # Load the network
    net = torch.load(f"models/learned_filters_{num_kernels}_{j}.pt", map_location=device)

    print("-------Before pruning-------")
    # Test the network before pruning
    pre_prune = test_suite(net, verbose=True)
    pre_prunes.append(pre_prune)

    # Define the pruning ratio for each layer, essentially, this is the proportion of kernel *pairs* for which one pair will be removed
    # In other words, if the pruning ratio is 1.0, then *at most* half of all the kernels for that layer will be removed
    # However, it is not *guaranteed* that half will be removed, since the similarity ordering can cause some kernels to be the most similar to multiple other kernels
    # thus, this kernel could be removed first, leading to the other pairs to have "already been pruned".
    # This is also why we have to manually compute the number of parameters removed
    prune_ratio = [0.35, 0.0, 0.0]

    # Prune the network
    net = prune_network(net, num_kernels, prune_ratio)

    # Count the number of non-zero weights
    nonzero_weights = count_nonzero_weights(net.state_dict(), num_kernels)
    new_kernel_num = nonzero_weights

    # Compute the new parameter count and the reduction
    new_param_count = compute_param_num(new_kernel_num[0], new_kernel_num[1], new_kernel_num[2])
    reduction_percentage = (1 - new_param_count / compute_param_num(num_kernels, num_kernels, num_kernels)) * 100
    reductions.append(reduction_percentage)

    print(f"\nRemoved {reduction_percentage:.2f}% of params")
    print("-------After pruning-------")

    # Test the network after pruning
    post_prune = test_suite(net, verbose=False)
    post_prunes.append(post_prune)

    # Print the results
    print(f"DaLiA DICE score: {post_prune[0]:.4f} ({post_prune[0] / pre_prune[0] * 100:.2f}% of original)")
    print(f"TROIKA DICE score: {post_prune[1]:.4f} ({post_prune[1] / pre_prune[1] * 100:.2f}% of original)")
    print(f"WESAD DICE score: {post_prune[2]:.4f} ({post_prune[2] / pre_prune[2] * 100:.2f}% of original)")
    print("=====================================")