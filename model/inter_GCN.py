import numpy as np

import torch
import torch.nn as nn

from .GraphConvolutionNetwork import GraphConvolutionNetwork
from .Element_Wise_Layer import Element_Wise_Layer

class inter_GCN(nn.Module):

    def __init__(self, interAdj, 
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80):

        super(inter_GCN, self).__init__()

        self.classNum = classNum
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.imageFeatureDim = imageFeatureDim

        self.interAdj = self.load_inter_Adj(interAdj)
        self.interGCN = GraphConvolutionNetwork(self.classNum, self.imageFeatureDim, self.intermediaDim, self.outputDim)

        self.fc = nn.Linear(2 * self.outputDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)

    def forward(self, input, semanticFeature):
        batchSize = input.size(0)

        interFeature = self.interGCN(semanticFeature, self.interAdj)                # (BatchSize, classNum, outputDim)
        output = torch.tanh(self.fc(torch.cat((interFeature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)), 1)))

        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)

        return result

    def load_inter_Adj(self, adj):
        Adj = adj.astype(np.float32)
        Adj = nn.Parameter(torch.from_numpy(Adj), requires_grad=True)
        return Adj