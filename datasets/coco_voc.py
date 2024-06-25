import os
import sys
import random
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image
from pycocotools.coco import COCO
from turtle import color
from email.mime import base
import xml.dom.minidom
from xml.dom.minidom import parse

sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

coco_new_id = [1,2,3,6,7,9,16,17,18,19,20,21,44,62]
voc_new_id = [14,1,6,5,18,3,2,7,11,12,16,9,4,8]
category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

class COCO_VOC(data.Dataset):

    def __init__(self, mode,
                 coco_image_dir, coco_anno_path, coco_labels_path,
                 voc_image_dir, voc_anno_path, voc_labels_path,
                 input_transform=None):

        assert mode in ('train', 'train34', 'train1', 'train2', 'val1', 'val2', 'val3', 'val4')
        self.mode = mode
        self.input_transform = input_transform
        print(self.mode)
        self.coco_root = coco_image_dir
        self.coco = COCO(coco_anno_path)
        self.coco_ids = list(self.coco.imgs.keys())

        self.voc_img_names  = []
        with open(voc_anno_path, 'r') as f:
             self.voc_img_names = f.readlines()
        self.voc_img_dir = voc_image_dir

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 means label exist)
        with open('./data/coco/category.json','r') as coco_load_category:
            self.coco_category_map = json.load(coco_load_category)
        self.coco_labels = []
        for i in range(len(self.coco_ids)):
            coco_img_id = self.coco_ids[i]
            coco_ann_ids = self.coco.getAnnIds(imgIds=coco_img_id)
            coco_target = self.coco.loadAnns(coco_ann_ids)
            self.coco_labels.append(getLabelVector(getCategoryList(coco_target), self.coco_category_map))
        self.coco_labels = np.array(self.coco_labels)
        self.coco_labels[self.coco_labels == 0] = -1
        self.new_coco_labels = self.coco_labels[:, :14]
        
        # labels : numpy.ndarray, shape->(len(self.img_names), 20)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.voc_labels = []
        for name in self.voc_img_names:
            voc_label_file = os.path.join(voc_labels_path,name[:-1]+'.xml')
            voc_label_vector = np.zeros(20)
            voc_DOMTree = xml.dom.minidom.parse(voc_label_file)
            voc_root = voc_DOMTree.documentElement
            voc_objects = voc_root.getElementsByTagName('object')  
            for obj in voc_objects:
                if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                    continue
                voc_tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                voc_label_vector[int(category_info[voc_tag])] = 1.0
            self.voc_labels.append(voc_label_vector)
        self.voc_labels = np.array(self.voc_labels).astype(np.int)
        self.voc_labels[self.voc_labels == 0] = -1
        self.new_voc_labels = self.voc_labels[:, :14]

        for i in range(0, 14):
            self.new_coco_labels[:, i] = self.coco_labels[:, self.coco_category_map[str(coco_new_id[i])]-1]
            self.new_voc_labels[:, i] = self.voc_labels[:, voc_new_id[i]]
            
        mask_coco = np.zeros(self.new_coco_labels.shape)
        mask_coco[:, :9] = 1
        mask_voc = np.zeros(self.new_voc_labels.shape)
        mask_voc[:, 5:] = 1

        self.intra1Labels = self.new_coco_labels * mask_coco
        self.intra1Labels = np.where(self.intra1Labels==-0, 0, self.intra1Labels)
        self.intra2Labels = self.new_voc_labels * mask_voc
        self.intra2Labels = np.where(self.intra2Labels==-0, 0, self.intra2Labels)

        self.labels = np.concatenate((self.intra1Labels, self.intra2Labels), axis=0)

        mask_voc_34 = np.zeros(self.new_voc_labels.shape)
        mask_voc_34[:, :9] = 1
        mask_coco_34 = np.zeros(self.new_coco_labels.shape)
        mask_coco_34[:, 5:] = 1
        
        self.intra3Labels = self.new_voc_labels * mask_voc_34
        self.intra3Labels = np.where(self.intra3Labels==-0, 0, self.intra3Labels)
        self.intra4Labels = self.new_coco_labels * mask_coco_34
        self.intra4Labels = np.where(self.intra4Labels==-0, 0, self.intra4Labels)

        self.labels34 = np.concatenate((self.intra3Labels, self.intra4Labels), axis=0)
        

    def __getitem__(self, index):
        if self.mode == 'train':
            if index < self.coco_labels.shape[0]:
                coco_img_id = self.coco_ids[index]
                coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
                coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
                if self.input_transform:
                    coco_input = self.input_transform(coco_input)
                return index, coco_input, self.intra1Labels[index], self.intra1Labels[index]
            else:
                idd = index - self.coco_labels.shape[0]
                voc_name = self.voc_img_names[idd][:-1]+'.jpg'
                voc_input = Image.open(os.path.join(self.voc_img_dir, voc_name)).convert('RGB')
                if self.input_transform:
                    voc_input = self.input_transform(voc_input)
                return index, voc_input, self.intra2Labels[idd], self.intra2Labels[idd]
        elif self.mode == 'train34':
            if  index < self.voc_labels.shape[0]:
                voc_name = self.voc_img_names[index][:-1]+'.jpg'
                voc_input = Image.open(os.path.join(self.voc_img_dir, voc_name)).convert('RGB')
                if self.input_transform:
                    voc_input = self.input_transform(voc_input)
                return index, voc_input, self.intra3Labels[index], self.intra3Labels[index]
            else:
                idd = index - self.voc_labels.shape[0]
                coco_img_id = self.coco_ids[idd]
                coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
                coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
                if self.input_transform:
                    coco_input = self.input_transform(coco_input)
                return index, coco_input, self.intra4Labels[idd], self.intra4Labels[idd]
        elif self.mode == 'val1':
            coco_img_id = self.coco_ids[index]
            coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
            coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
            if self.input_transform:
                coco_input = self.input_transform(coco_input)
            return index, coco_input, self.new_coco_labels[index], self.new_coco_labels[index]
        elif self.mode == 'val2':
            voc_name = self.voc_img_names[index][:-1]+'.jpg'
            voc_input = Image.open(os.path.join(self.voc_img_dir, voc_name)).convert('RGB')
            if self.input_transform:
                voc_input = self.input_transform(voc_input)
            return index, voc_input, self.new_voc_labels[index], self.new_voc_labels[index]      
        elif self.mode == 'val3':
            voc_name = self.voc_img_names[index][:-1]+'.jpg'
            voc_input = Image.open(os.path.join(self.voc_img_dir, voc_name)).convert('RGB')
            if self.input_transform:
                voc_input = self.input_transform(voc_input)
            return index, voc_input, self.new_voc_labels[index], self.new_voc_labels[index]    
        elif self.mode == 'val4':
            coco_img_id = self.coco_ids[index]
            coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
            coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
            if self.input_transform:
                coco_input = self.input_transform(coco_input)
            return index, coco_input, self.new_coco_labels[index], self.new_coco_labels[index]

    def __len__(self):
        if self.mode == 'train':
            return len(self.voc_img_names) + len(self.coco_ids)
        elif self.mode == 'train34':
            return len(self.voc_img_names) + len(self.coco_ids)
        elif self.mode == 'val1':
            return len(self.coco_ids)
        elif self.mode == 'val2':
            return len(self.voc_img_names)
        elif self.mode == 'val3':
            return len(self.voc_img_names)
        elif self.mode == 'val4':
            return len(self.coco_ids)
        

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
