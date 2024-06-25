import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionNetwork(nn.Module):

    def __init__(self, classNum, imageFeatureDim=1024, intermediaDim=512, outputDim=1024):

        super(GraphConvolutionNetwork, self).__init__()

        self.classNum = classNum
        self.imageFeatureDim = imageFeatureDim
        self.intermediaDim = intermediaDim
        self.outputDim = outputDim

        self.gc1 = GraphConvolution(imageFeatureDim, intermediaDim)
        self.gc2 = GraphConvolution(intermediaDim, outputDim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, input, adj):
        x = self.gc1(input, adj)                                        # (Batchsize, classNum, intermediaDim)
        x = self.relu(x)                                                # (Batchsize, classNum, intermediaDim)
        x = self.dropout(x)                                             # (Batchsize, classNum, intermediaDim)
        x = self.gc2(x, adj)                                            # (Batchsize, classNum, outputDim)
        return x
