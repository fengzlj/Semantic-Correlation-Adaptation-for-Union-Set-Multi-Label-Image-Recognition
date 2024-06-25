import numpy as np

import torch
import torch.nn as nn

from .GraphConvolutionNetwork import GraphConvolutionNetwork
from .Element_Wise_Layer import Element_Wise_Layer

class intra_GCN_ILRB(nn.Module):

    def __init__(self, intraAdj1, intraAdj2,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, isAlphaLearnable=True):
        
        super(intra_GCN_ILRB, self).__init__()

        self.classNum = classNum
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.imageFeatureDim = imageFeatureDim

        self.intraAdj1 = self.load_intra_Adj(intraAdj1)
        self.intraAdj2 = self.load_intra_Adj(intraAdj2)
        self.intraGCN = GraphConvolutionNetwork(self.classNum, self.imageFeatureDim, self.intermediaDim, self.outputDim)
        
        self.fc = nn.Linear(3 * self.imageFeatureDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)
        
        self.alpha = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isAlphaLearnable)
        
    def forward(self, input, semanticFeature, target=None):
        
        batchSize = input.size(0)

        intraFeature1 = self.intraGCN(semanticFeature, self.intraAdj1)              # (BatchSize, classNum, outputDim)
        intraFeature2 = self.intraGCN(semanticFeature, self.intraAdj2)              # (BatchSize, classNum, outputDim)
        intraFeature = torch.cat((intraFeature1, intraFeature2), 2)                 # (BatchSize, classNum, 2*outputDim)

        output = torch.tanh(self.fc(torch.cat((intraFeature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)), 1)))

        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)

        if target is None:
            return result

        self.alpha.data.clamp_(min=0, max=1)
        
        # Instance-level Mixup
        # coef ~ alpha
        coef, mixedTarget = self.mixupLabel(target, torch.flip(target, dims=[0]), self.alpha)
        coef = coef.unsqueeze(-1).repeat(1, 1, self.outputDim)
        mixedSemanticFeature = coef * semanticFeature + (1-coef) * torch.flip(semanticFeature, dims=[0])
        # Predict Category
        mixedFeature_1 = self.intraGCN(mixedSemanticFeature, self.intraAdj1)
        mixedFeature_2 = self.intraGCN(mixedSemanticFeature, self.intraAdj2) 
        mixedFeature = torch.cat((mixedFeature_1, mixedFeature_2), 2)
        mixedOutput = torch.tanh(self.fc(torch.cat((mixedFeature.view(batchSize * self.classNum, -1),
                                                    mixedSemanticFeature.view(-1, self.imageFeatureDim)), 1)))
        mixedOutput = mixedOutput.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult = self.classifiers(mixedOutput)                          # (batchSize, classNum)

        return result, mixedResult, mixedTarget

    def mixupLabel(self, label1, label2, alpha):

        matrix = torch.ones_like(label1).cuda()
        matrix[(label1 == 0) & (label2 == 1)] = alpha

        return matrix, matrix * label1 + (1-matrix) * label2

    def load_intra_Adj(self, adj):
        Adj = adj.astype(np.float32)
        Adj = nn.Parameter(torch.from_numpy(Adj), requires_grad=True)
        return Adj