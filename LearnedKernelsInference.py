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

#Set seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#Progress bar
from tqdm import tqdm

device = "cuda" #device to use


base = "PPG Data" #Base directory for the PPG data
subdirs = [ #sub dirs that contain each PPG dataset
"new_PPG_DaLiA_test/processed_dataset",
"new_PPG_DaLiA_train/processed_dataset",
"TROIKA_channel_1/processed_dataset",
"WESAD_all/processed_dataset"]


X_train = np.load(os.path.join(base, subdirs[1], "scaled_ppgs.npy"))
Y_train = np.load(os.path.join(base, subdirs[1], "seg_labels.npy"))

#The rest of these datasets are test
DaLiA_X = np.load(os.path.join(base, subdirs[0], "scaled_ppgs.npy"))
DaLiA_Y = np.load(os.path.join(base, subdirs[0], "seg_labels.npy"))

TROIKA_X = np.load(os.path.join(base, subdirs[2], "scaled_ppgs.npy"))
TROIKA_Y = np.load(os.path.join(base, subdirs[2], "seg_labels.npy"))

WESAD_X = np.load(os.path.join(base, subdirs[3], "scaled_ppgs.npy"))
WESAD_Y = np.load(os.path.join(base, subdirs[3], "seg_labels.npy"))

def test_model(net, X, Y):
    """
    Test the given network on the provided data.

    Args:
        net (nn.Module): The trained model.
        X (list): The list of data samples.
        Y (list): The list of corresponding labels.

    Returns:
        float: The DICE score of the model's predictions.
    """
    dtype = net.state_dict()['w1'].dtype # Get the data type of the model
    device = net.state_dict()['w1'].device # Get the device of the model
    data = X
    labels = Y

    preds = [] # To store the model predictions
    true = [] # To store the true labels

    for i in range(len(data)):
        x = data[i]
        x = (x - np.mean(x) ) /np.std(x) # Normalize the data

        y_pred = net(torch.tensor(x, dtype=dtype, device=device).view(1, -1)) # Predictions

        # Apply a Savitzky-Golay filter to the predictions and convert to binary form
        preds += list(np.float32(savgol_filter(y_pred.detach().cpu().float().numpy()[0], 151, 3) > 0.5))
        true += list(labels[i]) # Concatenate labels as list

    return f1_score(preds, true) # Return the F1 score, which in this case, is equivalent to DICE score

def test_suite(net, verbose=True):
    """
    Test the given network on all the datasets and return and/or print the DICE scores

    Args:
        net (nn.Module): The trained model.
        verbose (bool, optional): Whether to print the DICE scores. Defaults to True.

    Returns:
        list: The DICE scores for each dataset.
    """
    results = [0, 0, 0] # Will store the F1 scores for each dataset

    # Test on the DaLiA dataset
    results[0] = test_model(net, DaLiA_X, DaLiA_Y)
    if verbose:
        print("DaLiA DICE score: %.4f" % results[0])

    # Test on the TROIKA dataset
    results[1] = test_model(net, TROIKA_X, TROIKA_Y)
    if verbose:
        print("TROIKA DICE score: %.4f" % results[1])

    # Test on the WESAD dataset
    results[2] = test_model(net, WESAD_X, WESAD_Y)

    if verbose:
        print("WESAD DICE score: %.4f" % results[2])

    return results

filter_nums = [4, 8, 16, 24, 32, 64, 128, 256, 512] #number of filters to train models for
folds = 10 #number of folds to use for cross validation
save_dir = "models" #directory to save models to
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

test_results = {}  # will store the results of each model

# Loop through different numbers of filters
for filter_num in filter_nums:
    print(f"Testing kernel with {filter_num} filters...")
    # Loop through different versions of the model
    pbar = tqdm(range(0, folds))
    results = []
    for fold in pbar:  # loop through all the folds
        # Load the model
        net = torch.load(os.path.join(save_dir, f"learned_filters_{filter_num}_{fold}.pt"), map_location=device)

        # Test the model
        results.append(test_suite(net, verbose=False))

        # Update the progress bar
        pbar.set_description(f"DaLiA: %.4f, TROIKA: %.4f, WESAD: %.4f" % tuple(np.mean(results, axis=0)))

    test_results[filter_num] = np.transpose(
        results)  # transpose the results so that the rows are the datasets and the columns are the folds

#save results
with open(os.path.join(save_dir, "test_results.pkl"), "wb") as f:
    pickle.dump(test_results, f)

    