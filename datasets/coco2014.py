from email.mime import base
import os
import sys
from turtle import color
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets

from pycocotools.coco import COCO

import matplotlib.pyplot as plt


class COCO2014(data.Dataset):

    def __init__(self, mode,
                 image_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0, 
                 divide_label_proportion=-1, cross_proportion=0.0):
        # # sum coco, changed, divided dataset
        # print(label_proportion)
        # print(divide_label_proportion)
        # print(baseline_data_id)

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion
        self.divide_label_proportion = divide_label_proportion
        self.cross_proportion = cross_proportion

        self.root = image_dir
        self.coco = COCO(anno_path)
        self.ids = list(self.coco.imgs.keys())
     
        with open('/data2/wangxinyu/HCP-MLR-PL/data/coco/category.json','r') as load_category:
            self.category_map = json.load(load_category)

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 mea   ns label exist)
        self.labels = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            self.labels.append(getLabelVector(getCategoryList(target), self.category_map))
        self.labels = np.array(self.labels)
        self.labels[self.labels == 0] = -1

        # np.save('/data2/wangxinyu/HCP-MLR-PL/txt/sst_d2_graph/Labels.npy', np.array(self.labels))

        # # changedLabels : numpy.ndarray, shape->(len(coco), 80)
        # # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1.0:
            print('Changing label proportion...')
            self.changedLabels = changeLabelProportion(self.labels, self.label_proportion)

        # dividedLabels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.intra1Labels = self.labels
        self.intra2Labels = self.labels
        self.crossLabels = self.labels
        self.dividedLabels = self.labels
        if divide_label_proportion != -1 and divide_label_proportion != 1:
            print('Dividing label Matrix...')
            self.intra1Labels, self.intra2Labels, self.crossLabels, self.dividedLabels = divideLabelMatrix(self.labels, self.divide_label_proportion)

        # dividedLabels_cross : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.intra1Labels = self.labels
        self.intra2Labels = self.labels
        self.crossLabels = self.labels
        self.dividedLabels_cross = self.labels
        if cross_proportion != -1.0 and cross_proportion != 0.0:
            print(cross_proportion)
            print('Dividing label Matrix and Making label Matrix cross...')
            self.intra1Labels, self.intra2Labels, self.crossLabels, self.dividedLabels_cross = divideLabelMatrix_cross(self.labels, self.divide_label_proportion, self.cross_proportion)

        # np.save('/data2/wangxinyu/HCP-MLR-PL/dividedLabels.npy', np.array(self.dividedLabels))


    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        return index, input, self.dividedLabels_cross[index], self.labels[index]
        # return index, input, self.dividedLabels[index], self.labels[index]
        # return index, input, self.labels[index], self.mask_1, self.mask_2, self.labels[index]

    def __len__(self):
        return len(self.ids)

# =============================================================================
# Help Functions
# =============================================================================
def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)

def getLabelVector(categories, category_map):
    label = np.zeros(80)
    for c in categories:
        label[category_map[str(c)]-1] = 1.0
    return label

def getLabel(mode):

    assert mode in ('train', 'val')

    from utils.dataloader import get_data_path
    train_dir, train_anno, train_label, \
    test_dir, test_anno, test_label = get_data_path('COCO2014')

    if mode == 'train':
        image_dir, anno_path = train_dir, train_anno
    else:
        image_dir, anno_path = test_dir, test_anno

    coco = datasets.CocoDetection(root=image_dir, annFile=anno_path)
    with open('/data2/wangxinyu/HCP-MLR-PL/data/coco/category.json', 'r') as load_category:
        category_map = json.load(load_category)

    labels = []
    for i in range(len(coco)):
        labels.append(getLabelVector(getCategoryList(coco[i][1]), category_map))
    labels = np.array(labels).astype(np.float64)

    np.save('/data2/wangxinyu/HCP-MLR-PL/data/coco/{}_label_vectors.npy'.format(mode), labels)


def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label

def divideLabelMatrix_cross(labels, divide_label_proportion, cross_proportion):
    # N random, C positive_class
    # Set Random Seed

    np.random.seed(0)

    c_pos_sum = np.sum(labels, axis=0)
    c_sort_id = np.argsort(-c_pos_sum)
    n_rand_id = np.arange(0, labels.shape[0], 1)
    np.random.shuffle(n_rand_id)

    mask1 = np.zeros(labels.shape)
    mask2 = np.zeros(labels.shape)
    mask3 = np.zeros(labels.shape)
    mask = np.zeros(labels.shape)

    n_matrix = []
    c_matrix = []
    for i in range(0, divide_label_proportion):
        n_matrix.append(labels.shape[0]//divide_label_proportion)
        c_matrix.append(labels.shape[1]//divide_label_proportion)
    n_matrix[divide_label_proportion-1] += labels.shape[0] - (labels.shape[0]//divide_label_proportion)*divide_label_proportion
    c_matrix[divide_label_proportion-1] += labels.shape[1] - (labels.shape[1]//divide_label_proportion)*divide_label_proportion

    now_n = 0
    now_c = 0
    for i in range(0, divide_label_proportion):
        end_n = now_n + n_matrix[i]
        end_c = now_c + c_matrix[i]
        mask[int(now_n):int(end_n), c_sort_id[int(now_c):int(end_c)]] = 1
        if i == 0:
            mask1[int(now_n):int(end_n), c_sort_id[int(now_c):int(end_c)]] = 1
        elif i == 1:
            mask2[int(now_n):int(end_n), c_sort_id[int(now_c):int(end_c)]] = 1
        now_n = end_n
        now_c = end_c

    # cross-domain
    cross_begin_n = labels.shape[0] * (1 - cross_proportion) / 2
    cross_begin_c = labels.shape[1] * (1 - cross_proportion) / 2
    cross_n = labels.shape[0] * cross_proportion
    cross_c = labels.shape[1] * cross_proportion
    mask[int(cross_begin_n):int(cross_begin_n+cross_n), :] = 1
    mask[:, c_sort_id[int(cross_begin_c):int(cross_begin_c+cross_c)]] = 1
    mask1[int(cross_begin_n):int(cross_begin_n+cross_n), c_sort_id[0:int(cross_begin_c+cross_c)]] = 1
    mask1[0:int(cross_begin_n+cross_n), c_sort_id[int(cross_begin_c):int(cross_begin_c+cross_c)]] = 1
    mask2[int(cross_begin_n):int(cross_begin_n+cross_n), c_sort_id[int(cross_begin_c):labels.shape[1]]] = 1
    mask2[int(cross_begin_n):labels.shape[0], c_sort_id[int(cross_begin_c):int(cross_begin_c+cross_c)]] = 1
    mask3[int(cross_begin_n):int(cross_begin_n+cross_n), :] = 1

    mask = mask[n_rand_id]
    mask1 = mask1[n_rand_id]
    mask2 = mask2[n_rand_id]
    mask3 = mask3[n_rand_id]

    label1 = labels * mask1
    label2 = labels * mask2
    label3 = labels * mask3
    label = labels * mask

    label1 = np.where(label1==-0, 0, label1)
    label2 = np.where(label2==-0, 0, label2)
    label3 = np.where(label3==-0, 0, label3)
    label = np.where(label==-0, 0, label)

    assert label1.shape == labels.shape
    assert label2.shape == labels.shape
    assert label3.shape == labels.shape
    assert label.shape == labels.shape
    # np.save('/data2/wangxinyu/HCP-MLR-PL/txt/c_sort_id.npy', c_sort_id)
    # np.save('/data2/wangxinyu/HCP-MLR-PL/txt/n_rand_id.npy', n_rand_id)

    return label1, label2, label3, label

def divideLabelMatrix(labels, divide_label_proportion):
    # N random, C positive_class
    # Set Random Seed

    np.random.seed(0)

    c_pos_sum = np.sum(labels, axis=0)
    c_sort_id = np.argsort(-c_pos_sum)
    n_rand_id = np.arange(0, labels.shape[0], 1)
    np.random.shuffle(n_rand_id)

    mask1 = np.zeros(labels.shape)
    mask2 = np.zeros(labels.shape)
    mask3 = np.zeros(labels.shape)
    mask = np.zeros(labels.shape)

    n_matrix = []
    c_matrix = []
    for i in range(0, divide_label_proportion):
        n_matrix.append(labels.shape[0]//divide_label_proportion)
        c_matrix.append(labels.shape[1]//divide_label_proportion)
    n_matrix[divide_label_proportion-1] += labels.shape[0] - (labels.shape[0]//divide_label_proportion)*divide_label_proportion
    c_matrix[divide_label_proportion-1] += labels.shape[1] - (labels.shape[1]//divide_label_proportion)*divide_label_proportion

    now_n = 0
    now_c = 0
    for i in range(0, divide_label_proportion):
        end_n = now_n + n_matrix[i]
        end_c = now_c + c_matrix[i]
        mask[int(now_n):int(end_n), c_sort_id[int(now_c):int(end_c)]] = 1
        if i == 0:
            mask1[int(now_n):int(end_n), c_sort_id[int(now_c):int(end_c)]] = 1
        elif i == 1:
            mask2[int(now_n):int(end_n), c_sort_id[int(now_c):int(end_c)]] = 1
        now_n = end_n
        now_c = end_c

    mask = mask[n_rand_id]
    mask1 = mask1[n_rand_id]
    mask2 = mask2[n_rand_id]
    mask3 = mask3[n_rand_id]

    label1 = labels * mask1
    label2 = labels * mask2
    label3 = labels * mask3
    label = labels * mask

    label1 = np.where(label1==-0, 0, label1)
    label2 = np.where(label2==-0, 0, label2)
    label3 = np.where(label3==-0, 0, label3)
    label = np.where(label==-0, 0, label)

    assert label1.shape == labels.shape
    assert label2.shape == labels.shape
    assert label3.shape == labels.shape
    assert label.shape == labels.shape
    # np.save('/data2/wangxinyu/HCP-MLR-PL/txt/c_sort_id.npy', c_sort_id)
    # np.save('/data2/wangxinyu/HCP-MLR-PL/txt/n_rand_id.npy', n_rand_id)

    return label1, label2, label3, label

def getCoOccurrenceLabel(mode):

    assert mode in ('train', 'val')

    if mode == 'train':
        label_path = '/data2/wangxinyu/HCP-MLR-PL/data/coco/train_label_vectors.npy'
    else:
        label_path = '/data2/wangxinyu/HCP-MLR-PL/data/coco/val_label_vectors.npy'

    labels = np.load(label_path).astype(np.float64)

    coOccurrenceLabel = np.zeros((labels.shape[0], sum([i for i in range(80)])), dtype=np.float64)
    for index in range(labels.shape[0]):
        correlationMatrix = labels[index][:, np.newaxis] * labels[index][np.newaxis, :]

        index_ = 0
        for i in range(80):
            for j in range(i + 1, 80):
                if correlationMatrix[i, j] > 0:
                    coOccurrenceLabel[index, index_] = 1
                index_ += 1

    np.save('/data2/wangxinyu/HCP-MLR-PL/data/coco/{}_co-occurrence_label_vectors.npy'.format(mode), coOccurrenceLabel)

def getPairIndexes(labels):

    res = []
    for index in range(labels.shape[0]):
        tmp = []
        for i in range(labels.shape[1]):
            if labels[index, i] > 0:
                tmp += np.where(labels[:, i] > 0)[0].tolist()

        tmp = set(tmp)
        tmp.discard(index)
        res.append(np.array(list(tmp)))

    return res


# self.mask_1 = np.ones(self.labels.shape[1], dtype=int)
        # self.mask_2 = np.ones(self.labels.shape[1], dtype=int)
        # c_pos_sum = np.sum(self.labels, axis=0)
        # c_sort_id = np.argsort(-c_pos_sum)
        # for i in range(0, self.labels.shape[1]):
        #     if i < self.labels.shape[1] - self.divide_label_proportion:
        #         self.mask_2[c_sort_id[i]] = 0
        #     else:
        #         self.mask_1[c_sort_id[i]] = 0

        # # sum coco, changed, divided dataset
        # if mode == 'train':
        #     total_num = self.labels.shape[0] * self.labels.shape[1]
        #     tags = ['1', '-1', '0']
        #     colors = ['red', 'blue', 'orange']

        #     coco_pos_num = np.sum(self.labels==1)
        #     coco_neg_num = np.sum(self.labels==-1)
        #     coco_zero_num = np.sum(self.labels==0)
        #     coco_num = [coco_pos_num/total_num*100, coco_neg_num/total_num*100, coco_zero_num/total_num*100]
        #     plt.figure(0)
        #     plt.pie(coco_num, labels=tags, colors=colors, autopct='%1.1f%%')
        #     plt.axis('equal')
        #     plt.title('coco_num')
        #     plt.savefig('/data2/wangxinyu/HCP-MLR-PL/coco_num.png')

        #     changed_pos_num = np.sum(self.changedLabels==1)
        #     changed_neg_num = np.sum(self.changedLabels==-1)
        #     changed_zero_num = np.sum(self.changedLabels==0)
        #     changed_num = [changed_pos_num/total_num*100, changed_neg_num/total_num*100, changed_zero_num/total_num*100]
        #     plt.figure(1)
        #     plt.pie(changed_num, labels=tags, colors=colors, autopct='%1.1f%%')
        #     plt.axis('equal')
        #     plt.title('changed_num')
        #     plt.savefig('/data2/wangxinyu/HCP-MLR-PL/changed_num.png')

        #     divided_pos_num = np.sum(self.dividedLabels==1)
        #     divided_neg_num = np.sum(self.dividedLabels==-1)
        #     divided_zero_num = np.sum(self.dividedLabels==0)
        #     divided_num = [divided_pos_num/total_num*100, divided_neg_num/total_num*100, divided_zero_num/total_num*100]
        #     plt.figure(2)
        #     plt.pie(divided_num, labels=tags, colors=colors, autopct='%1.1f%%')
        #     plt.axis('equal')
        #     plt.title('divided_num')
        #     plt.savefig('/data2/wangxinyu/HCP-MLR-PL/divided_num.png')


        # def divideLabelMatrix(labels, divide_label_proportion, baseline_data_id):

#     # Set Random Seed

#     np.random.seed(0)

#     c_pos_sum = np.sum(labels, axis=0)
#     c_sort_id = np.argsort(-c_pos_sum)

#     mask = np.random.random(labels.shape[0])
#     mask[mask < (divide_label_proportion / labels.shape[1])] = 1
#     mask[mask < 1] = 0
#     mask = np.tile(mask, (labels.shape[1], 1))
#     mask = mask.T
    
#     bid = labels.shape[1] - divide_label_proportion
#     for i in range(bid, labels.shape[1]):
#         mask[:, c_sort_id[i]] = np.where(mask[:, c_sort_id[i]]>0, 0, 1)
    
#     if baseline_data_id == 1:
#         for i in range(bid, labels.shape[1]):
#             mask[:, c_sort_id[i]] = 0
#     elif baseline_data_id == 2:
#         for i in range(0, bid):
#             mask[:, c_sort_id[i]] = 0

#     label = labels * mask
#     label = np.where(label==-0, 0, label)

#     assert label.shape == labels.shape

#     return label