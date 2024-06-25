import numpy as np

import torch
import torch.nn as nn

from .backbone.resnet import resnet101

class ResNet(nn.Module):

    def __init__(self, 
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3):
        
        super(ResNet, self).__init__()

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

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.outputDim, self.classNum)

    def forward(self, input):

        batchSize = input.size(0)

        featureMap = self.backbone(input)                                   # (BatchSize, 2048, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)                     # (BatchSize, imgFeatureDim, imgSize, imgSize)

        # Predict Category
        output = self.pool(featureMap)                                      # (BatchSize, imgFeatureDim, 1, 1)
        output = output.contiguous().view(batchSize, -1)                    # (BatchSize, imgFeatureDim)

        result = self.fc(output)                                            # (BatchSize, classNum)
        
        return result