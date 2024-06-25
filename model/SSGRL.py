import numpy as np

import torch
import torch.nn as nn

from .backbone.resnet import resnet101
from .SemanticDecoupling import SemanticDecoupling

class SSGRL(nn.Module):

    def __init__(self, wordFeatures,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300):
        
        super(SSGRL, self).__init__()

        self.backbone = resnet101()

        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)
        
        self.classNum = classNum
        
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim

        self.wordFeatures = self.load_features(wordFeatures)

        self.semanticDecoupling = SemanticDecoupling(self.classNum, self.imageFeatureDim, self.wordFeatureDim, intermediaDim=self.intermediaDim)

        self.posFeature = None

    def forward(self, input):
        
        batchSize = input.size(0)

        featureMap = self.backbone(input)                               # (BatchSize, Channel, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)                 # (BatchSize,imgFeatureDim, imgSize, imgSize)
        
        semanticFeature = self.semanticDecoupling(featureMap, self.wordFeatures)[0]

        return semanticFeature
    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum-1):
            res[0] += [index for i in range(classNum-index-1)]
            res[1] += [i for i in range(index+1, classNum)]
        return res
        
    def updateFeature(self, feature, target, exampleNum):

        if self.posFeature is None:
            self.posFeature = torch.zeros((self.classNum, exampleNum, feature.size(-1))).cuda()

        feature = feature.detach().clone()
        for c in range(self.classNum):
            posFeature = feature[:, c][target[:, c] == 1]
            self.posFeature[c] = torch.cat((posFeature, self.posFeature[c]), dim=0)[:exampleNum]

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)