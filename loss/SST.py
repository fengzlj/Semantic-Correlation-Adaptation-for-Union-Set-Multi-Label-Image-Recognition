import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def getIntraPseudoLabel(intraCoOccurrence, target, margin=0.50):
    """
    Shape of intraCoOccurrence : (batchSize, classNum ** 2)
    Shape of target : (batchSize, classNum)
    """
    batchSize, classNum = target.size(0), target.size(1)
    probCoOccurrence = torch.sigmoid(intraCoOccurrence)

    indexStart, indexEnd, pseudoLabel = 0, 0, torch.zeros((batchSize, classNum, classNum)).cuda()
    for i in range(classNum):
        pseudoLabel[:, i, i] = 1
        indexStart = indexEnd
        indexEnd += classNum-i-1
        pseudoLabel[:, i, i+1:] = probCoOccurrence[:, indexStart:indexEnd]
        pseudoLabel[:, i+1:, i] = probCoOccurrence[:, indexStart:indexEnd]

    target_ = target.detach().clone()
    target_[target_ != 1] = 0
    pseudoLabel = torch.sum(pseudoLabel * target_.view(batchSize, 1, classNum).repeat(1, classNum, 1), dim=2)
    pseudoLabel = pseudoLabel / torch.clamp(torch.sum(target_, dim=1), min=1).view(batchSize, 1).repeat(1, classNum)
    pseudoLabel = torch.clamp(pseudoLabel, min=-1, max=1)

    pseudoLabel[pseudoLabel >= margin] = 1
    pseudoLabel[pseudoLabel <= margin] = 0

    pseudoLabel[target != 0] = 0
    pseudoLabel = target.detach().clone() + pseudoLabel

    return pseudoLabel


def getInterPseudoLabel(feature, target, posFeature, margin=0.50):
    """
    Shape of feature : (BatchSize, classNum, featureDim)
    Shape of posfeature : (classNum, exampleNum, featureDim)
    Shape of psedudoLabel, target : (BatchSize, ClassNum)
    """

    batchSize, classNum, featureDim = feature.size()
    exampleNum, pseudoLabel, cos = posFeature.size(1), target.detach().clone(), torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    # you can use torch.repeat_interleave when you use pytorch >= 1.1.0
    target = target.cpu()
    feature1 = feature[target == 0].view(-1, 1, featureDim).repeat(1, exampleNum, 1)  # (targetNum, exampleNum, featureDim)
    feature2 = posFeature[np.where(target == 0)[1]]                                   # (targetNum, exampleNum, featureDim)
    posDistance = torch.mean(cos(feature1, feature2), dim=1)

    posDistance[posDistance >= margin] = 1
    posDistance[posDistance <= margin] = 0

    pseudoLabel[target == 0] = posDistance

    return pseudoLabel

def getInterAdjMatrix(target):
    """
    Shape of target : (BatchSize, ClassNum)
    """

    target_ = target.detach().clone().permute(1, 0)
    target_[target_ != 1] = 0
    adjMatrix = target_.unsqueeze(1).repeat(1, target.size(0), 1) * target_.unsqueeze(2).repeat(1, 1, target.size(0))
    eyeMatrix = torch.eye(target.size(0)).unsqueeze(0).repeat(target.size(1), 1, 1).cuda()
    adjMatrix = torch.clamp(adjMatrix + eyeMatrix, max=1, min=0)

    return adjMatrix


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


class AsymmetricLoss(nn.Module):

    def __init__(self, margin=0, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduce=None, size_average=None):
        super(AsymmetricLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Get positive and negative mask
        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            input_sigmoid_neg = (input_sigmoid_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = input_sigmoid_pos * positive_mask + input_sigmoid_neg * negative_mask
            one_sided_gamma = self.gamma_pos * positive_mask + self.gamma_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, one_sided_gamma)

            loss *= one_sided_weight

        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss


class intraAsymmetricLoss(nn.Module):

    def __init__(self, classNum, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduce=None, size_average=None):
        super(intraAsymmetricLoss, self).__init__()

        self.classNum = classNum
        self.concatIndex = self.getConcatIndex(classNum)

        self.reduce = reduce
        self.size_average = size_average

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, \sum_{i=1}^{classNum-1}{i})
        Shape of target: (BatchSize, classNum)
        """
        target = target.cpu().data.numpy()
        target1, target2 = target[:, self.concatIndex[0]], target[:, self.concatIndex[1]]
        target1, target2 = (target1 > 0).astype(np.float), (target2 > 0).astype(np.float)
        target = torch.Tensor(target1 * target2).cuda()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            input_sigmoid_neg = (input_sigmoid_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = target * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = (1 - target) * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = input_sigmoid_pos * target + input_sigmoid_neg * (1 - target)
            one_sided_gamma = self.gamma_pos * target + self.gamma_neg * (1 - target)
            one_sided_weight = torch.pow(1 - prob, one_sided_gamma)

            loss *= one_sided_weight

        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res


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

        """
        target_ = target.detach().clone()
        target_[target != 1] = 0
        positiveTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]
        negativeTarget = 1 - positiveTarget
        negativeTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
        negativeTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            positive_loss = (1 - distance)[positiveTarget == 1]
            negative_loss = (1 + distance)[negativeTarget == 1]

            if positive_loss.size(0) == 0:
                if self.size_average:
                    return torch.mean(negative_loss) if negative_loss.size(0) != 0 else 0 * torch.mean(negative_loss)
                return torch.sum(negative_loss)

            if positive_loss.size(0) != 0:
                negative_loss = torch.cat((torch.index_select(negative_loss, 0, torch.randperm(negative_loss.size(0))[:2 * positive_loss.size(0)].cuda()),
                                           torch.sort(negative_loss, descending=True)[0][:positive_loss.size(0)]), 0)
            if self.size_average:
                return torch.mean(torch.cat((positive_loss, negative_loss), 0)) if positive_loss.size(0) != 0 else torch.mean(negative_loss)
            return torch.sum(torch.cat((positive_loss, negative_loss), 0)) if positive_loss.size(0) != 0 else torch.sum(negative_loss)

        return distance
        """

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res

# =============================================================================
# Check Code
# =============================================================================
def main():

    target = [[1, 1, -1, 1],
              [-1, 1, 1, 1],
              [1, 1, -1, -1]]
    target = torch.Tensor(target).cuda()
    adjMatrix = getInterAdjMatrix(target)
    print(adjMatrix)


if __name__ == "__main__":
    main()
