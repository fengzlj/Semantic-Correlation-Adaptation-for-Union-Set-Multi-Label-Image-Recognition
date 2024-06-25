import os
import PIL
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.vg import VG
from datasets.coco2014 import COCO2014
from datasets.voc2007 import VOC2007
from datasets.vg_coco import VG_COCO
from datasets.coco_voc import COCO_VOC
from datasets.voc_vg import VOC_VG

from config import prefixPathCOCO, prefixPathVG, prefixPathVOC2007, prefixPathOpenImage

def get_graph_file(labels):

    graph = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float)

    for index in range(labels.shape[0]):
        indexs = np.where(labels[index] == 1)[0]
        for i in indexs:
            for j in indexs:
                graph[i, j] += 1
    
    for i in range(labels.shape[1]):
        if graph[i, i] == 0:
            graph[i] = 0
        else:
            graph[i] /= graph[i, i]

    np.nan_to_num(graph)

    return graph

def get_inter_graph_file(WordFile):

    graph = np.zeros((WordFile.shape[0], WordFile.shape[0]), dtype=np.float)
    WordFile_T = torch.Tensor(WordFile)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-9)

    for i in range(0, WordFile.shape[0]):
        for j in range(0, WordFile.shape[0]):
            graph[i, j] = cos(WordFile_T[i], WordFile_T[j])

    return graph

def get_word_file(args):

    if args.dataset == 'COCO2014':
        WordFilePath = './data/coco/vectors.npy'
        WordFile = np.load(WordFilePath)
        
    elif args.dataset == 'VG':
        WordFilePath = './data/vg/vg_200_vector.npy'
        WordFile = np.load(WordFilePath)

    elif args.dataset == 'VOC2007':
        WordFilePath = './data/voc_devkit/VOC2007/voc07_vector.npy'
        WordFile = np.load(WordFilePath)

    elif args.dataset == 'COCO_VOC':
        WordFilePath = './data/coco/vectors.npy'
        coco_new_id = [1,2,3,6,7,9,16,17,18,19,20,21,44,62]

        WordFile = np.load(WordFilePath)
        WordFile = WordFile[coco_new_id]

    elif args.dataset == 'VG_COCO':
        vg_new_id = [5,84,18,147,104,108,102,135,69,93,82,142,96,100,130,123,118,149,132,120,195,71,112,146,158,63,121,101,85,180,157,36,160,197,156,99,98,186]
        WordFilePath = './data/vg/vg_200_vector.npy'

        WordFile = np.load(WordFilePath)
        WordFile = WordFile[vg_new_id]

    elif args.dataset == 'VOC_VG':
        voc_new_id = [5,3,1,4,6,9,12,17,18,16,7,8,11,2,14]
        WordFilePath = './data/voc_devkit/VOC2007/voc07_vector.npy'

        WordFile = np.load(WordFilePath)
        WordFile = WordFile[voc_new_id]
    
    return WordFile

def get_data_path(dataset):

    if dataset == 'VOC_VG':

        voc_train_dir, voc_train_anno, voc_train_label = os.path.join(prefixPathVOC2007, 'JPEGImages'), os.path.join(prefixPathVOC2007, 'ImageSets/Main/trainval.txt'), os.path.join(prefixPathVOC2007, 'Annotations')
        vg_train_dir, vg_train_anno, vg_train_label = os.path.join(prefixPathVG, 'VG_100K'), './data/vg/train_list_500.txt', './data/vg/vg_category_200_labels_index.json'
        voc_test_dir, voc_test_anno, voc_test_label = os.path.join(prefixPathVOC2007, 'JPEGImages'), os.path.join(prefixPathVOC2007, 'ImageSets/Main/test.txt'), os.path.join(prefixPathVOC2007, 'Annotations')
        vg_test_dir, vg_test_anno, vg_test_label = os.path.join(prefixPathVG, 'VG_100K'), './data/vg/test_list_500.txt', './data/vg/vg_category_200_labels_index.json'

        return voc_train_dir, voc_train_anno, voc_train_label, \
               vg_train_dir, vg_train_anno, vg_train_label, \
               voc_test_dir, voc_test_anno, voc_test_label, \
               vg_test_dir, vg_test_anno, vg_test_label

    if dataset == 'VG_COCO':

        coco_train_dir, coco_train_anno, coco_train_label = os.path.join(prefixPathCOCO, 'train2014'), os.path.join(prefixPathCOCO, 'annotations/instances_train2014.json'), './data/coco/train_label_vectors.npy'
        vg_train_dir, vg_train_anno, vg_train_label = os.path.join(prefixPathVG, 'VG_100K'), './data/vg/train_list_500.txt', './data/vg/vg_category_200_labels_index.json'
        coco_test_dir, coco_test_anno, coco_test_label = os.path.join(prefixPathCOCO, 'val2014'), os.path.join(prefixPathCOCO, 'annotations/instances_val2014.json'), './data/coco/val_label_vectors.npy'
        vg_test_dir, vg_test_anno, vg_test_label = os.path.join(prefixPathVG, 'VG_100K'), './data/vg/test_list_500.txt', './data/vg/vg_category_200_labels_index.json'

        return vg_train_dir, vg_train_anno, vg_train_label, \
               coco_train_dir, coco_train_anno, coco_train_label, \
               coco_test_dir, coco_test_anno, coco_test_label, \
               vg_test_dir, vg_test_anno, vg_test_label
    
    if dataset == 'COCO_VOC':

        coco_train_dir, coco_train_anno, coco_train_label = os.path.join(prefixPathCOCO, 'train2014'), os.path.join(prefixPathCOCO, 'annotations/instances_train2014.json'), './data/coco/train_label_vectors.npy'
        voc_train_dir, voc_train_anno, voc_train_label = os.path.join(prefixPathVOC2007, 'JPEGImages'), os.path.join(prefixPathVOC2007, 'ImageSets/Main/trainval.txt'), os.path.join(prefixPathVOC2007, 'Annotations')
        coco_test_dir, coco_test_anno, coco_test_label = os.path.join(prefixPathCOCO, 'val2014'), os.path.join(prefixPathCOCO, 'annotations/instances_val2014.json'), './data/coco/val_label_vectors.npy'
        voc_test_dir, voc_test_anno, voc_test_label = os.path.join(prefixPathVOC2007, 'JPEGImages'), os.path.join(prefixPathVOC2007, 'ImageSets/Main/test.txt'), os.path.join(prefixPathVOC2007, 'Annotations')

        return coco_train_dir, coco_train_anno, coco_train_label, \
               voc_train_dir, voc_train_anno, voc_train_label, \
               coco_test_dir, coco_test_anno, coco_test_label, \
               voc_test_dir, voc_test_anno, voc_test_label

    if dataset == 'COCO2014':
        prefixPath = prefixPathCOCO
        train_dir, train_anno, train_label = os.path.join(prefixPath, 'train2014'), os.path.join(prefixPath, 'annotations/instances_train2014.json'), './data/coco/train_label_vectors.npy'
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'val2014'), os.path.join(prefixPath, 'annotations/instances_val2014.json'), './data/coco/val_label_vectors.npy'

    elif dataset == 'VG':
        prefixPath = prefixPathVG
        train_dir, train_anno, train_label = os.path.join(prefixPath, 'VG_100K'), './data/vg/train_list_500.txt', './data/vg/vg_category_200_labels_index.json'
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'VG_100K'), './data/vg/test_list_500.txt', './data/vg/vg_category_200_labels_index.json'

    elif dataset == 'VOC2007':
        prefixPath = prefixPathVOC2007
        train_dir, train_anno, train_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath, 'ImageSets/Main/trainval.txt'), os.path.join(prefixPath, 'Annotations')
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath, 'ImageSets/Main/test.txt'), os.path.join(prefixPath, 'Annotations')

    return train_dir, train_anno, train_label, \
           test_dir, test_anno, test_label

def get_data_loader(args): 
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    randomCropList = [transforms.RandomCrop(Size) for Size in [640, 576, 512, 448, 384, 320]] if args.scaleSize == 640 else \
                     [transforms.RandomCrop(Size) for Size in [512, 448, 384, 320, 256]]
    train_data_transform = transforms.Compose([transforms.Resize((args.scaleSize, args.scaleSize), interpolation=PIL.Image.BICUBIC),
                                               transforms.RandomChoice(randomCropList),
                                               transforms.Resize((args.cropSize, args.cropSize), interpolation=PIL.Image.BICUBIC),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
    
    test_data_transform = transforms.Compose([transforms.Resize((args.cropSize, args.cropSize), interpolation=PIL.Image.BICUBIC),
                                              transforms.ToTensor(),
                                              normalize])
    
    

    if args.dataset == 'COCO2014':  
        train_dir, train_anno, train_label, \
        test_dir, test_anno, test_label = get_data_path(args.dataset)

        print("==> Loading COCO2014...")
        # # ResNet-test(no need train)
        # train_set = COCO2014('train',
        #                      train_dir, train_anno, train_label,
        #                      input_transform=train_data_transform)
        # # ResNet-test                     
        # test_set = COCO2014('val',
        #                      test_dir, test_anno, test_label,
        #                      input_transform=test_data_transform)
        
        # # divided-labels
        # train_set = COCO2014('train',
        #                      train_dir, train_anno, train_label,
        #                      input_transform=train_data_transform, 
        #                      divide_label_proportion=args.divide )
        # test_set = COCO2014('val',
        #                     test_dir, test_anno, test_label,
        #                     input_transform=test_data_transform)

        # cross-labels
        train_set = COCO2014('train',
                             train_dir, train_anno, train_label,
                             input_transform=train_data_transform, 
                             divide_label_proportion=args.divide,
                             cross_proportion=args.cross )
        test_set = COCO2014('val',
                            test_dir, test_anno, test_label,
                            input_transform=test_data_transform)
        train_loader = DataLoader(dataset=train_set,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        test_loader = DataLoader(dataset=test_set,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
                             
        return train_loader, test_loader
        
        # # part-label
        # train_set = COCO2014('train',
        #                      train_dir, train_anno, train_label,
        #                      input_transform=train_data_transform, label_proportion=args.prob)
        # test_set = COCO2014('val',
        #                     test_dir, test_anno, test_label,
        #                     input_transform=test_data_transform)

    elif args.dataset == 'VG':
        train_dir, train_anno, train_label, \
        test_dir, test_anno, test_label = get_data_path(args.dataset)
        print("==> Loading VG...")
        # train_set = VG('train',
        #                train_dir, train_anno, train_label,
        #                input_transform=train_data_transform, label_proportion=args.prob)
        # test_set = VG('val',
        #               test_dir, test_anno, test_label,
        #               input_transform=test_data_transform)

        # train_set = VG('train',
        #                train_dir, train_anno, train_label,
        #                input_transform=train_data_transform, 
        #                divide_label_proportion=args.divide)
        # test_set = VG('val',
        #               test_dir, test_anno, test_label,
        #               input_transform=test_data_transform)

        # cross-labels
        train_set = VG('train',
                        train_dir, train_anno, train_label,
                        input_transform=train_data_transform, 
                        divide_label_proportion=args.divide,
                        cross_proportion=args.cross )
        test_set = VG('val',
                       test_dir, test_anno, test_label,
                       input_transform=test_data_transform)
        
        train_loader = DataLoader(dataset=train_set,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        test_loader = DataLoader(dataset=test_set,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
                             
        return train_loader, test_loader
    
    elif args.dataset == 'VOC2007':
        train_dir, train_anno, train_label, \
        test_dir, test_anno, test_label = get_data_path(args.dataset)
        print("==> Loading VOC2007...")

        # partial-labels
        # train_set = VOC2007('train',
        #                     train_dir, train_anno, train_label,
        #                     input_transform=train_data_transform, label_proportion=args.prob)
        # test_set = VOC2007('val',
        #                    test_dir, test_anno, test_label,
        #                    input_transform=test_data_transform)
        
        # cross-labels
        train_set = VOC2007('train',
                             train_dir, train_anno, train_label,
                             input_transform=train_data_transform, 
                             divide_label_proportion=args.divide,
                             cross_proportion=args.cross )
        test_set = VOC2007('val',
                            test_dir, test_anno, test_label,
                            input_transform=test_data_transform)
        train_loader = DataLoader(dataset=train_set,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        test_loader = DataLoader(dataset=test_set,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
                             
        return train_loader, test_loader
                       
    elif args.dataset == 'VG_COCO':
        vg_train_dir, vg_train_anno, vg_train_label, \
        coco_train_dir, coco_train_anno, coco_train_label, \
        coco_test_dir, coco_test_anno, coco_test_label, \
        vg_test_dir, vg_test_anno, vg_test_label = get_data_path(args.dataset)
        print("==> Loading VG_COCO...")
        train_set = VG_COCO('train',
                            vg_train_dir, vg_train_anno, vg_train_label,
                            coco_train_dir, coco_train_anno, coco_train_label,
                            input_transform=train_data_transform)
        # train_set34 = VG_COCO('train34',
        #                     vg_train_dir, vg_train_anno, vg_train_label,
        #                     coco_train_dir, coco_train_anno, coco_train_label,
        #                     input_transform=train_data_transform)
        train_set1 = VG_COCO('train1',
                            vg_train_dir, vg_train_anno, vg_train_label,
                            coco_train_dir, coco_train_anno, coco_train_label,
                            input_transform=train_data_transform)
        train_set2 = VG_COCO('train2',
                            vg_train_dir, vg_train_anno, vg_train_label,
                            coco_train_dir, coco_train_anno, coco_train_label,
                            input_transform=train_data_transform)
        test_set1 = VG_COCO('val1',
                           vg_test_dir, vg_test_anno, vg_test_label,
                           coco_test_dir, coco_test_anno, coco_test_label,
                           input_transform=test_data_transform)
        # test_set3 = VG_COCO('val3',
        #                    vg_test_dir, vg_test_anno, vg_test_label,
        #                    coco_test_dir, coco_test_anno, coco_test_label,
        #                    input_transform=test_data_transform)
        test_set2 = VG_COCO('val2',
                           vg_test_dir, vg_test_anno, vg_test_label,
                           coco_test_dir, coco_test_anno, coco_test_label,
                           input_transform=test_data_transform)
        # test_set4 = VG_COCO('val4',
        #                    vg_test_dir, vg_test_anno, vg_test_label,
        #                    coco_test_dir, coco_test_anno, coco_test_label,
        #                    input_transform=test_data_transform)


        train_loader = DataLoader(dataset=train_set,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        # train_loader34 = DataLoader(dataset=train_set34,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=True)
        train_loader1 = DataLoader(dataset=train_set1,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        train_loader2 = DataLoader(dataset=train_set2,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        test_loader1 = DataLoader(dataset=test_set1,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
        test_loader2 = DataLoader(dataset=test_set2,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                             shuffle=False)
        # test_loader3 = DataLoader(dataset=test_set3,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=False)
        # test_loader4 = DataLoader(dataset=test_set4,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                      shuffle=False)
        
        # return train_loader1, test_loader1                 
        # return train_loader2, test_loader2                 
        return train_loader, test_loader1, test_loader2
        # return train_loader34, test_loader3, test_loader4
    
    elif args.dataset == 'COCO_VOC':
        coco_train_dir, coco_train_anno, coco_train_label, \
        voc_train_dir, voc_train_anno, voc_train_label, \
        coco_test_dir, coco_test_anno, coco_test_label, \
        voc_test_dir, voc_test_anno, voc_test_label = get_data_path(args.dataset)
        print("==> Loading COCO_VOC...")
        # train_set = COCO_VOC('train',
        #                     coco_train_dir, coco_train_anno, coco_train_label,
        #                     voc_train_dir, voc_train_anno, voc_train_label,
        #                     input_transform=train_data_transform)
        # test_set1 = COCO_VOC('val1',
        #                     coco_train_dir, coco_train_anno, coco_train_label,
        #                     voc_test_dir, voc_test_anno, voc_test_label,
        #                     input_transform=train_data_transform)
        # test_set2 = COCO_VOC('val2',
        #                     coco_train_dir, coco_train_anno, coco_train_label,
        #                     voc_test_dir, voc_test_anno, voc_test_label,
        #                     input_transform=train_data_transform)
        
        train_set34 = COCO_VOC('train34',
                            coco_train_dir, coco_train_anno, coco_train_label,
                            voc_train_dir, voc_train_anno, voc_train_label,
                            input_transform=train_data_transform)
        test_set3 = COCO_VOC('val3',
                            coco_train_dir, coco_train_anno, coco_train_label,
                            voc_test_dir, voc_test_anno, voc_test_label,
                            input_transform=train_data_transform)
        test_set4 = COCO_VOC('val4',
                            coco_train_dir, coco_train_anno, coco_train_label,
                            voc_test_dir, voc_test_anno, voc_test_label,
                            input_transform=train_data_transform)


        # train_loader = DataLoader(dataset=train_set,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=True)
        # test_loader1 = DataLoader(dataset=test_set1,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=False)
        # test_loader2 = DataLoader(dataset=test_set2,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=False)
        train_loader34 = DataLoader(dataset=train_set34,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        test_loader3 = DataLoader(dataset=test_set3,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
        test_loader4 = DataLoader(dataset=test_set4,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
                       
        # return train_loader, test_loader1, test_loader2
        return train_loader34, test_loader3, test_loader4

    elif args.dataset == 'VOC_VG':
        voc_train_dir, voc_train_anno, voc_train_label, \
        vg_train_dir, vg_train_anno, vg_train_label, \
        voc_test_dir, voc_test_anno, voc_test_label, \
        vg_test_dir, vg_test_anno, vg_test_label = get_data_path(args.dataset)
        print("==> Loading VOC_VG...")
        # train_set = VOC_VG('train',
        #                     voc_train_dir, voc_train_anno, voc_train_label,
        #                     vg_train_dir, vg_train_anno, vg_train_label,
        #                     input_transform=train_data_transform)
        # test_set1 = VOC_VG('val1',
        #                     voc_test_dir, voc_test_anno, voc_test_label,
        #                     vg_test_dir, vg_test_anno, vg_test_label,
        #                     input_transform=train_data_transform)
        # test_set2 = VOC_VG('val2',
        #                     voc_test_dir, voc_test_anno, voc_test_label,
        #                     vg_test_dir, vg_test_anno, vg_test_label,
        #                     input_transform=train_data_transform)
        
        train_set34 = VOC_VG('train34',
                            voc_train_dir, voc_train_anno, voc_train_label,
                            vg_train_dir, vg_train_anno, vg_train_label,
                            input_transform=train_data_transform)
        test_set3 = VOC_VG('val3',
                            voc_test_dir, voc_test_anno, voc_test_label,
                            vg_test_dir, vg_test_anno, vg_test_label,
                            input_transform=train_data_transform)
        test_set4 = VOC_VG('val4',
                            voc_test_dir, voc_test_anno, voc_test_label,
                            vg_test_dir, vg_test_anno, vg_test_label,
                            input_transform=train_data_transform)


        # train_loader = DataLoader(dataset=train_set,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=True)
        # test_loader1 = DataLoader(dataset=test_set1,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=False)
        # test_loader2 = DataLoader(dataset=test_set2,
        #                         num_workers=args.workers,
        #                         batch_size=args.batchSize,
        #                         pin_memory=True,
        #                         drop_last=True,
        #                         shuffle=False)
        train_loader34 = DataLoader(dataset=train_set34,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=True)
        test_loader3 = DataLoader(dataset=test_set3,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
        test_loader4 = DataLoader(dataset=test_set4,
                                num_workers=args.workers,
                                batch_size=args.batchSize,
                                pin_memory=True,
                                drop_last=True,
                                shuffle=False)
                       
        # return train_loader, test_loader1, test_loader2
        return train_loader34, test_loader3, test_loader4


