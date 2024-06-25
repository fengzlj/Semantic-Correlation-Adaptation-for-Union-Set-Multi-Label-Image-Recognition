import numpy as np

import torch
import torch.nn as nn

from .GraphConvolutionNetwork import GraphConvolutionNetwork
from .Element_Wise_Layer import Element_Wise_Layer

class intra_GCN(nn.Module):

    def __init__(self, intraAdj1, intraAdj2,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80):
        
        super(intra_GCN, self).__init__()

        self.classNum = classNum
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.imageFeatureDim = imageFeatureDim

        self.intraAdj1 = self.load_intra_Adj(intraAdj1)
        self.intraAdj2 = self.load_intra_Adj(intraAdj2)
        self.intraGCN = GraphConvolutionNetwork(self.classNum, self.imageFeatureDim, self.intermediaDim, self.outputDim)
        
        self.fc = nn.Linear(3 * self.imageFeatureDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)

    def forward(self, input, semanticFeature):
        
        batchSize = input.size(0)

        intraFeature1 = self.intraGCN(semanticFeature, self.intraAdj1)              # (BatchSize, classNum, outputDim)
        intraFeature2 = self.intraGCN(semanticFeature, self.intraAdj2)              # (BatchSize, classNum, outputDim)
        intraFeature = torch.cat((intraFeature1, intraFeature2), 2)                 # (BatchSize, classNum, 2*outputDim)

        output = torch.tanh(self.fc(torch.cat((intraFeature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)), 1)))

        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)

        return result

    def load_intra_Adj(self, adj):
        Adj = adj.astype(np.float32)
        Adj = nn.Parameter(torch.from_numpy(Adj), requires_grad=True)
        return Adj