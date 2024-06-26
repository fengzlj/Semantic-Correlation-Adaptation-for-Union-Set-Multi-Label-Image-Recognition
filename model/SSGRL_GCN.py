import numpy as np

import torch
import torch.nn as nn

from .backbone.resnet import resnet101
from .GraphConvolutionNetwork import GraphConvolutionNetwork, GraphConvolution
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer


class SSGRL_GCN(nn.Module):

    def __init__(self, intraAdj1, intraAdj2, interAdj, wordFeatures,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3):

        super(SSGRL_GCN, self).__init__()

        self.backbone = resnet101()

        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)

        self.classNum = classNum
        self.timeStep = timeStep

        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim
        
        self.intraAdj1 = self.load_intra_Adj(intraAdj1)
        self.intraAdj2 = self.load_intra_Adj(intraAdj2)
        self.interAdj = self.load_inter_Adj(interAdj)
        self.wordFeatures = self.load_features(wordFeatures)     

        self.SemanticDecoupling = SemanticDecoupling(self.classNum, self.imageFeatureDim, self.wordFeatureDim, intermediaDim=self.intermediaDim)
        self.intraGCN = GraphConvolutionNetwork(self.classNum, self.imageFeatureDim, self.intermediaDim, self.outputDim)
        self.interGCN = GraphConvolution(2*outputDim, outputDim)

        self.fc = nn.Linear(2 * self.imageFeatureDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)

    def forward(self, input):

        batchSize = input.size(0)

        featureMap = self.backbone(input)                                            # (BatchSize, Channel, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)                              # (BatchSize, imgFeatureDim, imgSize, imgSize)

        semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)[0]  # (BatchSize, classNum, imgFeatureDim)
        intraFeature1 = self.intraGCN(semanticFeature, self.intraAdj1)               # (BatchSize, classNum1, outputDim)
        intraFeature2 = self.intraGCN(semanticFeature, self.intraAdj2)               # (BatchSize, classNum2, outputDim)
        intraFeature = torch.cat((intraFeature1, intraFeature2), 2)                  # (BatchSize, classNum, 2*outputDim)
        feature = self.interGCN(intraFeature, self.interAdj)                         # (BatchSize, classNum, outputDim)
        # Predict Category
        output = torch.tanh(self.fc(torch.cat((feature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)),1)))

        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)                                            # (BatchSize, classNum)


        return result, semanticFeature


    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum-1):
            res[0] += [index for i in range(classNum-index-1)]
            res[1] += [i for i in range(index+1, classNum)]
        return res

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)
    
    def load_inter_Adj(self, adj):
        Adj = adj.astype(np.float32)
        Adj = nn.Parameter(torch.from_numpy(Adj), requires_grad=True)
        return Adj

    def load_intra_Adj(self, adj):
        Adj = adj.astype(np.float32)
        Adj = nn.Parameter(torch.from_numpy(Adj), requires_grad=False)
        return Adj
# =============================================================================
# Help Functions
# =============================================================================
