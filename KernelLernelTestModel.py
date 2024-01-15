import torch
from LearnedKernelModel import *

#Math
from sklearn.metrics import f1_score, r2_score
import numpy as np
from scipy.signal import savgol_filter

from LearnedKernelData import *

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
