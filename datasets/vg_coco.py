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

sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

coco_new_id = [1,2,3,4,5,6,7,8,9,15,16,17,18,19,20,21,22,23,24,25,27,28,38,41,42,44,47,51,52,53,59,62,65,70,81,84,85,86]
vg_new_id = [5,84,18,147,104,108,102,135,69,93,82,142,96,100,130,123,118,149,132,120,195,71,112,146,158,63,121,101,85,180,157,36,160,197,156,99,98,186]

class VG_COCO(data.Dataset):

    def __init__(self, mode,
                 vg_image_dir, vg_anno_path, vg_labels_path,
                 coco_image_dir, coco_anno_path, coco_labels_path,
                 input_transform=None):

        assert mode in ('train', 'train34', 'train1', 'train2', 'val1', 'val2', 'val3', 'val4')
        self.mode = mode
        self.input_transform = input_transform

        self.vg_img_dir = vg_image_dir
        self.vg_imgName_path = vg_anno_path
        self.vg_img_names = open(self.vg_imgName_path, 'r').readlines()

        self.coco_root = coco_image_dir
        self.coco = COCO(coco_anno_path)
        self.coco_ids = list(self.coco.imgs.keys())

        # labels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.vg_labels_path = vg_labels_path
        _ = json.load(open(self.vg_labels_path, 'r'))
        self.vg_labels = np.zeros((len(self.vg_img_names), 200)).astype(np.int) - 1
        for i in range(len(self.vg_img_names)):
            self.vg_labels[i][_[self.vg_img_names[i][:-1]]] = 1
        self.new_vg_labels = self.vg_labels[:, :38]

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
        self.new_coco_labels = self.coco_labels[:, :38]

        for i in range(0, 38):
            self.new_coco_labels[:, i] = self.coco_labels[:, self.coco_category_map[str(coco_new_id[i])]-1]
            self.new_vg_labels[:, i] = self.vg_labels[:, vg_new_id[i]]
            
        mask_coco = np.zeros(self.new_coco_labels.shape)
        mask_coco[:, :23] = 1
        mask_vg = np.zeros(self.new_vg_labels.shape)
        mask_vg[:, 15:] = 1

        self.intra1Labels = self.new_coco_labels * mask_coco
        self.intra1Labels = np.where(self.intra1Labels==-0, 0, self.intra1Labels)
        self.intra2Labels = self.new_vg_labels * mask_vg
        self.intra2Labels = np.where(self.intra2Labels==-0, 0, self.intra2Labels)

        mask_vg_34 = np.zeros(self.new_vg_labels.shape)
        mask_vg_34[:, :23] = 1
        mask_coco_34 = np.zeros(self.new_coco_labels.shape)
        mask_coco_34[:, 15:] = 1
        
        self.intra3Labels = self.new_vg_labels * mask_vg_34
        self.intra3Labels = np.where(self.intra3Labels==-0, 0, self.intra3Labels)
        self.intra4Labels = self.new_coco_labels * mask_coco_34
        self.intra4Labels = np.where(self.intra4Labels==-0, 0, self.intra4Labels)

        self.labels = np.concatenate((self.intra3Labels, self.intra4Labels), axis=0)
        

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
                idd = index-self.coco_labels.shape[0]
                vg_name = self.vg_img_names[idd][:-1]
                vg_input = Image.open(os.path.join(self.vg_img_dir, vg_name)).convert('RGB')
                if self.input_transform:
                    vg_input = self.input_transform(vg_input)
                return index, vg_input, self.intra2Labels[idd], self.intra2Labels[idd]
        elif self.mode == 'train34':
            if  index < self.vg_labels.shape[0]:
                vg_name = self.vg_img_names[index][:-1]
                vg_input = Image.open(os.path.join(self.vg_img_dir, vg_name)).convert('RGB')
                if self.input_transform:
                    vg_input = self.input_transform(vg_input)
                return index, vg_input, self.intra3Labels[index], self.intra3Labels[index]
            else:
                idd = index-self.vg_labels.shape[0]
                coco_img_id = self.coco_ids[idd]
                coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
                coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
                if self.input_transform:
                    coco_input = self.input_transform(coco_input)
                return index, coco_input, self.intra4Labels[idd], self.intra4Labels[idd]
        elif self.mode == 'train1':
            coco_img_id = self.coco_ids[index]
            coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
            coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
            if self.input_transform:
                coco_input = self.input_transform(coco_input)
            return index, coco_input, self.intra1Labels[index], self.intra1Labels[index]
        elif self.mode == 'train2':
            vg_name = self.vg_img_names[index][:-1]
            vg_input = Image.open(os.path.join(self.vg_img_dir, vg_name)).convert('RGB')
            if self.input_transform:
                vg_input = self.input_transform(vg_input)
            return index, vg_input, self.intra2Labels[index], self.intra2Labels[index]
        elif self.mode == 'val1':
            coco_img_id = self.coco_ids[index]
            coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
            coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
            if self.input_transform:
                coco_input = self.input_transform(coco_input)
            return index, coco_input, self.new_coco_labels[index], self.new_coco_labels[index]
        elif self.mode == 'val2':
            vg_name = self.vg_img_names[index][:-1]
            vg_input = Image.open(os.path.join(self.vg_img_dir, vg_name)).convert('RGB')
            # if self.input_transform:
            vg_input = self.input_transform(vg_input)
            return index, vg_input, self.new_vg_labels[index], self.new_vg_labels[index]      
        elif self.mode == 'val3':
            vg_name = self.vg_img_names[index][:-1]
            vg_input = Image.open(os.path.join(self.vg_img_dir, vg_name)).convert('RGB')
            if self.input_transform:
                vg_input = self.input_transform(vg_input)
            return index, vg_input, self.new_vg_labels[index], self.new_vg_labels[index]    
        elif self.mode == 'val4':
            coco_img_id = self.coco_ids[index]
            coco_path = self.coco.loadImgs(coco_img_id)[0]['file_name']
            coco_input = Image.open(os.path.join(self.coco_root, coco_path)).convert('RGB')
            if self.input_transform:
                coco_input = self.input_transform(coco_input)
            return index, coco_input, self.new_coco_labels[index], self.new_coco_labels[index]

    def __len__(self):
        if self.mode == 'train':
            return len(self.vg_img_names) + len(self.coco_ids)
        elif self.mode == 'train34':
            return len(self.vg_img_names) + len(self.coco_ids)
        elif self.mode == 'train1':
            return len(self.coco_ids)
        elif self.mode == 'train2' :
            return len(self.vg_img_names)
        elif self.mode == 'val1':
            return len(self.coco_ids)
        elif self.mode == 'val2':
            return len(self.vg_img_names)
        elif self.mode == 'val3':
            return len(self.vg_img_names)
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
