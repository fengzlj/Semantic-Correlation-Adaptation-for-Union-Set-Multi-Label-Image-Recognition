import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def getInterPseudoLabel(feature, target, posFeature, margin=0.50):
    """
    Shape of feature : (BatchSize, classNum, featureDim)
    Shape of posfeature : (classNum, exampleNum, featureDim)
    Shape of psedudoLabel, target : (BatchSize, ClassNum)
    """
    batchsize, classNum, featureDim = feature.size()
    exampleNum, pseudoLabel, cos = posFeature.size(1), target.detach(), torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    # you can use torch.repeat_interleave when you use pytorch >= 1.1.0
    target = target.cpu()
    feature1 = feature[target == 0].view(-1, 1, featureDim).repeat(1, exampleNum, 1)  # (targetNum, exampleNum, featureDim)
    feature2 = posFeature[np.where(target == 0)[1]]
    posDistance = torch.mean(cos(feature1, feature2), dim=1)

    posDistance[posDistance >= margin] = 1
    posDistance[posDistance <= margin] = 0

    pseudoLabel[target == 0] = posDistance

    return pseudoLabel

class BCELoss(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELoss, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target):

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)

        loss = positive_mask * positive_loss + negative_mask * negative_loss

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.mean(loss)
            return torch.sum(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.sum(loss)
        return loss

class KLLoss(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None, co_lambda=0.1):
        super(KLLoss, self).__init__()

        self.margin = margin
        self.reduce = reduce
        self.size_average = size_average
        self.co_lambda = co_lambda
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)
    
    def forward(self, input1, input2, target, mixupepoch, mixedoutput, mixedtarget, pseudotarget, BCEW, epoch, generateLabelEpoch, forget_rate):
        loss_pick_1 = self.bce_loss(input1, target) 
        loss_pick_2 = 0 * self.bce_loss(input1, pseudotarget) if epoch < generateLabelEpoch else BCEW * self.bce_loss(input1, pseudotarget) 
        loss_pick_3 = 0 * loss_pick_1 if epoch < mixupepoch else self.bce_loss(mixedoutput, mixedtarget)
        loss_pick = ((loss_pick_1 + loss_pick_2 + loss_pick_3) * (1-self.co_lambda) + self.co_lambda * self.kl_loss(input1, input2) + self.co_lambda * self.kl_loss(input2, input1)).cpu() 
        loss_pick = torch.mean(loss_pick, dim=1)

        ind_sorted = np.argsort(loss_pick.data)
        loss_sorted = loss_pick[ind_sorted]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]
        loss = torch.mean(loss_pick[ind_update])

        return loss

    def kl_loss(self, input1, input2):
        kl = F.kl_div(F.log_softmax(input1, dim=1), F.softmax(input2, dim=1), reduce=False)

        return kl

    def bce_loss(self, input, target):
        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)

        bce = positive_mask * positive_loss + negative_mask * negative_loss

        return bce

        

class ContrastiveLoss(nn.Module):
    """
    Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, batchSize, reduce=None, size_average=None):
        super(ContrastiveLoss, self).__init__()

        self.batchSize = batchSize
        self.concatIndex = self.getConcatIndex(batchSize)

        self.reduce = reduce
        self.size_average = size_average

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum, featureDim)
        Shape of target: (BatchSize, classNum), Value range of target: (-1, 0, 1)
        """

        target_ = target.detach().clone()
        target_[target_ != 1] = 0
        pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        pos2negTarget = 1 - pos2posTarget
        pos2negTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
        pos2negTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

        target_ = -1 * target.detach().clone()
        target_[target_ != 1] = 0
        neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            pos2pos_loss = (1 - distance)[pos2posTarget == 1]
            pos2neg_loss = (1 + distance)[pos2negTarget == 1]
            neg2neg_loss = (1 + distance)[neg2negTarget == 1]

            if pos2pos_loss.size(0) != 0:
                if neg2neg_loss.size(0) != 0:
                    neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                              torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
                if pos2neg_loss.size(0) != 0:
                    if pos2neg_loss.size(0) != 0:    
                        pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                                  torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)

            loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)

            if self.size_average:
                return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
            return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())
 
        return distance

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res