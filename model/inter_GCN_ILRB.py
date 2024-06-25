import numpy as np

import torch
import torch.nn as nn

from .GraphConvolutionNetwork import GraphConvolutionNetwork
from .Element_Wise_Layer import Element_Wise_Layer

class inter_GCN_ILRB(nn.Module):

    def __init__(self, interAdj, 
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, isAlphaLearnable=True):

        super(inter_GCN_ILRB, self).__init__()

        self.classNum = classNum
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.imageFeatureDim = imageFeatureDim

        self.interAdj = self.load_inter_Adj(interAdj)
        self.interGCN = GraphConvolutionNetwork(self.classNum, self.imageFeatureDim, self.intermediaDim, self.outputDim)

        self.fc = nn.Linear(2 * self.outputDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)

        self.alpha = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isAlphaLearnable)

    def forward(self, input, semanticFeature, target=None):
        batchSize = input.size(0)

        interFeature = self.interGCN(semanticFeature, self.interAdj)                # (BatchSize, classNum, outputDim)
        output = torch.tanh(self.fc(torch.cat((interFeature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)), 1)))

        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)

        if target is None:
            return result

        self.alpha.data.clamp_(min=0, max=1)

        # Instance-level Mixup
        # coef ~ alpha
        coef, mixedTarget_1 = self.mixupLabel(target, torch.flip(target, dims=[0]), self.alpha)
        coef = coef.unsqueeze(-1).repeat(1, 1, self.outputDim)
        mixedSemanticFeature_1 = coef * semanticFeature + (1-coef) * torch.flip(semanticFeature, dims=[0])
        # Predict Category
        mixedFeature_1 = self.interGCN(mixedSemanticFeature_1, self.interAdj) 
        mixedOutput_1 = torch.tanh(self.fc(torch.cat((mixedFeature_1.view(batchSize * self.classNum, -1),
                                                      mixedSemanticFeature_1.view(-1, self.imageFeatureDim)), 1)))
        mixedOutput_1 = mixedOutput_1.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult_1 = self.classifiers(mixedOutput_1)                          # (batchSize, classNum)

        return result, mixedResult_1, mixedTarget_1

    def mixupLabel(self, label1, label2, alpha):

        matrix = torch.ones_like(label1).cuda()
        matrix[(label1 == 0) & (label2 == 1)] = alpha

        return matrix, matrix * label1 + (1-matrix) * label2

    def load_inter_Adj(self, adj):
        Adj = adj.astype(np.float32)
        Adj = nn.Parameter(torch.from_numpy(Adj), requires_grad=True)
        return Adj