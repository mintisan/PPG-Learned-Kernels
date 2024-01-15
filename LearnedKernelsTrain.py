import os #path/directory stuff
import pickle

#Deep learning
import torch
from LearnedKernelModel import *

#Math
from sklearn.metrics import f1_score, r2_score
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

from LearnedKernelData import *

from torchinfo import summary
model = LearnedFilters(8)
summary(model, input_size=(1, 1, 200))  # 第二个 1 是输入通道数，L 是输入数据的长度

from thop import profile
from thop import clever_format

model = LearnedFilters(8)
input = torch.randn(1, 1, 100)  # 假设 L 是输入数据的长度
flops, params = profile(model, inputs=(input, ))
flops, params = clever_format([flops, params], "%.3f")
print('FLOPs: ', flops)
print('Params: ', params)

num_kernels = 8  # 这是输出通道数
input_channels = 1  # 假设输入通道数是1
kernel_sizes = [192, 96, 64]  # 卷积核大小
bias = True  # 假设每层都有偏置项
L = 100

# 计算每层的参数内存
for kernel_size in kernel_sizes:
    param_memory = (input_channels * kernel_size * num_kernels + (1 if bias else 0) * num_kernels) * 4
    print(f"Param memory for kernel size {kernel_size}: {param_memory} bytes")

# 假设输入长度L是固定的，我们计算激活内存
# 注意：对于padding="same"，输出长度与输入长度相同
batch_size = 1  # 假设批处理大小为1
for kernel_size in kernel_sizes:
    output_height = L  # 因为padding="same"，所以输出长度与输入长度相同
    activation_memory = (batch_size * num_kernels * output_height) * 4
    print(f"Activation memory for kernel size {kernel_size}: {activation_memory} bytes")


### Setup
filter_nums = [4, 8, 16, 24, 32, 64, 128, 256, 512] #number of filters to train models for
folds = 10 #number of folds to use for cross validation
epochs = 512 #number of epochs to train for
lr = 0.01 #learning rate
wd = 1e-4 #weight decay
decay_range = [1.0, 0.2] #range of decay values to use for learning rate decay

save_dir = "models" #directory to save models to
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through different numbers of filters
for filter_num in filter_nums:
    # Loop through different versions of the model
    for fold in range(0, folds):
        # Initialize a new instance of the LearnedFilters class with the current number of filters
        net = LearnedFilters(filter_num).to(device)

        # Compute the total number of model parameters and print it
        params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters())])
        print(f"Training kernel with {filter_num} filters (fold {fold + 1})...")
        print("Num params: %i" % params)

        # Initialize the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

        # Initialize a linear learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=decay_range[0], end_factor=decay_range[1],
                                                      total_iters=epochs)

        # Initialize a progress bar for visualization
        pbar = tqdm(range(0, epochs))

        # Normalize the input data by subtracting the mean and dividing by the standard deviation
        x = copy.deepcopy(X_train)  # we deepcopy the data so we don't modify the original

        # normalize each signal
        for i in range(0, len(x)):
            x[i] = (x[i] - np.mean(x[i])) / np.std(x[i])

        # Convert the input and output data to PyTorch tensors and move them to the device
        x = torch.tensor(x, dtype=torch.float32, device=device).reshape(len(x), 1, 1920)
        y = torch.tensor(Y_train, dtype=torch.float32, device=device)

        loss_hist = []  # history of losses (used to update the progress bar primarily)

        # Train the model
        for step in pbar:
            # Zero out the gradients
            optimizer.zero_grad()

            # Initialize the total loss to 0
            total_loss = 0

            # Split the input data into smaller batches if the number of filters is greater than 32 to avoid running out of memory
            split = 32 if filter_num > 32 else 1
            for i in range(0, split):
                split_len = x.shape[0] // split
                out = net(x[i * split_len:(i + 1) * split_len])  # Forward pass
                loss = F.binary_cross_entropy(out.view(-1),
                                              y[i * split_len:(i + 1) * split_len].view(-1)) / split  # Compute the loss
                total_loss += loss.item()  # Add the loss to the total loss
                loss.backward()  # Backward pass

            optimizer.step()  # Take an optimizer step
            scheduler.step()  # Adjust the learning rate

            loss_hist.append(total_loss)  # Add the total loss to the loss history
            pbar.set_description(
                "Loss: %.5f" % np.mean(loss_hist[-256:]))  # Update the progress bar with the current loss

        # Save the model
        torch.save(net, os.path.join(save_dir, f"learned_filters_{filter_num}_{fold}.pt"))