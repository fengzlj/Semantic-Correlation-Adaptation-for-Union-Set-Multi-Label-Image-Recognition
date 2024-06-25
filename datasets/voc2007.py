import os
import sys
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))


import numpy as np
from PIL import Image
import xml.dom.minidom
from xml.dom.minidom import parse

import torch
import torch.utils.data as data

category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

class VOC2007(data.Dataset):

    def __init__(self, mode,
                 img_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0,
                 divide_label_proportion=-1, cross_proportion=-1):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion
        self.divide_label_proportion = divide_label_proportion
        self.cross_proportion = cross_proportion

        self.img_names  = []
        with open(anno_path, 'r') as f:
             self.img_names = f.readlines()
        self.img_dir = img_dir
        
        self.labels = []
        for name in self.img_names:
            label_file = os.path.join(labels_path,name[:-1]+'.xml')
            label_vector = np.zeros(20)
            DOMTree = xml.dom.minidom.parse(label_file)
            root = DOMTree.documentElement
            objects = root.getElementsByTagName('object')  
            for obj in objects:
                if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                    continue
                tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                label_vector[int(category_info[tag])] = 1.0
            self.labels.append(label_vector)

        # labels : numpy.ndarray, shape->(len(self.img_names), 20)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels = np.array(self.labels).astype(np.int)
        self.labels[self.labels == 0] = -1

        # changedLabels : numpy.ndarray, shape->(len(self.img_names), 20)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1:
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
        if cross_proportion != -1.0:
            print(cross_proportion)
            print('Dividing label Matrix and Making label Matrix cross...')
            self.intra1Labels, self.intra2Labels, self.crossLabels, self.dividedLabels_cross = divideLabelMatrix_cross(self.labels, self.divide_label_proportion, self.cross_proportion)

    def __getitem__(self, index):
        name = self.img_names[index][:-1]+'.jpg'
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        return index, input, self.dividedLabels_cross[index], self.labels[index]
        # return index, input, self.dividedLabels[index], self.labels[index]
        # return index, input, self.changedLabels[index], self.labels[index]

    def __len__(self):
        return len(self.img_names)


# =============================================================================
# Help Functions
# =============================================================================
def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label

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

    if cross_proportion != 0.0:
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