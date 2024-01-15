#Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedFilters(nn.Module):
    def __init__(self, num_kernels=24):
        super(LearnedFilters, self).__init__()
        self.conv1 = nn.Conv1d(1, num_kernels, 192, stride=1, padding="same", bias=True)
        self.conv2 = nn.Conv1d(1, num_kernels, 96, stride=1, padding="same", bias=True)
        self.conv3 = nn.Conv1d(1, num_kernels, 64, stride=1, padding="same", bias=True)

        self.w1 = torch.nn.Parameter(torch.zeros(num_kernels), requires_grad=True)  # these are learned weights for the kernels
        self.w2 = torch.nn.Parameter(torch.zeros(num_kernels), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.zeros(num_kernels), requires_grad=True)


    def forward(self, x):
        batch_size = x.shape[0]

        c1 = F.relu(F.relu(self.conv1(x))) * self.w1[None ,: ,None]
        c2 = F.relu(F.relu(self.conv2(x))) * self.w2[None ,: ,None]
        c3 = F.relu(F.relu(self.conv3(x))) * self.w3[None ,: ,None]

        aggregate = torch.cat([c1 ,c2 ,c3], dim=1)
        aggregate = aggregate.sum(dim=1).view(batch_size, -1)
        aggregate = torch.sigmoid(aggregate)

        return aggregate