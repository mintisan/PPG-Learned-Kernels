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


from KernelLernelTestModel import *


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

