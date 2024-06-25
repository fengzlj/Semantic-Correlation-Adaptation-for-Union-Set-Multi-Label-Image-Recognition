"""
Configuration file!
"""

import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Dataset Path
# =============================================================================
prefixPathCOCO = '/data2/dataset/MS-COCO_2014/'
prefixPathVG = '/data2/dataset/VG/'
prefixPathVOC2007 = '/data2/dataset/PASCAL/voc2007/VOCdevkit/VOC2007/'
prefixPathOpenImage = '/data2/datasets/OpenImageV6'
# =============================================================================

# ClassNum of Dataset
# =============================================================================
_ClassNum = {'COCO2014': 80,
             'VOC2007': 20,
             'VG': 200,
             'OpenImage': 300,
             'VG_COCO':38,
             'COCO_VOC':14,
             'VOC_VG':15,
            }
# =============================================================================


# Argument Parse
# =============================================================================
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse(mode):

    assert mode in ('ResNet', 'SSGRL', 'CST', 'CST-ILRB', 'ILRB')

    parser = argparse.ArgumentParser(description='HCP Multi-label Image Recognition with Partial Labels')

    # Basic Augments
    parser.add_argument('--post', type=str, default='', help='postname of save model')
    parser.add_argument('--printFreq', type=int, default='1000', help='number of print frequency (default: 1000)')

    parser.add_argument('--mode', type=str, default='SSGRL', choices=['ResNet', 'SSGRL', 'CST', 'CST-ILRB', 'ILRB'], help='mode of experiment (default: SST)')
    parser.add_argument('--dataset', type=str, default='COCO2014', choices=['VOC_VG', 'COCO_VOC', 'VG_COCO', 'COCO2014', 'VG', 'VOC2007', 'OpenImage'], help='dataset for training and testing')
    parser.add_argument('--prob', type=float, default=1.0, help='hyperparameter of label proportion (default: 1.0)')

    parser.add_argument('--pretrainedModel', type=str, default='None', help='path to pretrained model (default: None)')
    parser.add_argument('--resumeModel', type=str, default='None', help='path to resume model (default: None)')
    # parser.add_argument('--resumeModel_1', type=str, default='None', help='path to resume model (default: None)')
    # parser.add_argument('--resumeModel_2', type=str, default='None', help='path to resume model (default: None)')
    parser.add_argument('--evaluate', type=str2bool, default='False', help='whether to evaluate model (default: False)')

    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run (default: 20)')
    parser.add_argument('--startEpoch', type=int, default=0, help='manual epoch number (default: 0)')
    parser.add_argument('--stepEpoch', type=int, default=10, help='decend the lr in epoch number (default: 10)')

    parser.add_argument('--batchSize', type=int, default=8, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weightDecay', type=float, default=1e-4, help='weight decay (default: 0.0001)')

    parser.add_argument('--cropSize', type=int, default=448, help='size of crop image')
    parser.add_argument('--scaleSize', type=int, default=512, help='size of rescale image')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--cross', type=float, default=-1.0, help='how much label matrix cross(default: -1)')
    parser.add_argument('--co_lambda', type=float, default=0.0, help='parameter of co_lambda(default: 0.0)')
    parser.add_argument('--forget_rate', type=float, help='forget rate', default=0.2)
    parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type=float, default=1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')

    # Aguments for ResNet
    if mode == 'ResNet':
        parser.add_argument('--divide', type=int, default=-1, help='divide label matrix (default: -1)')
        # parser.add_argument('--baselineId', type=int, default=0, help='baseline model to choose corresponding (default: 0)')

    if mode == 'SSGRL':
        parser.add_argument('--divide', type=int, default=-1, help='divide label matrix (default: -1)')
        # parser.add_argument('--baselineId', type=int, default=0, help='baseline model to choose corresponding (default: 0)')
    
    if mode == 'CST':
        parser.add_argument('--divide', type=int, default=-1, help='divide label matrix (default: -1)')
        # parser.add_argument('--baselineId', type=int, default=0, help='baseline model to choose corresponding (default: 0)')
        parser.add_argument('--generateLabelEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')
        parser.add_argument('--pseudoBCEMargin', type=float, default=1.0, help='margin of pseudo bce loss (default: 1.0)')
        parser.add_argument('--pseudoBCEWeight', type=float, default=1.0, help='weight of pseudo bce loss (default: 1.0)')
        parser.add_argument('--pseudoExampleNumber', type=int, default=50, help='number of pseudo positive number (default: 50)')
        parser.add_argument('--pseudoDistanceWeight', type=float, default=1.0, help='weight of pseudo Distance loss (default: 1.0)')
    
    if mode == 'CST-ILRB':
        parser.add_argument('--divide', type=int, default=-1, help='divide label matrix (default: -1)')
        # parser.add_argument('--baselineId', type=int, default=0, help='baseline model to choose corresponding (default: 0)')
        parser.add_argument('--generateLabelEpoch', type=int, default=5, help='when to generate pseudo label (default: 5)')
        parser.add_argument('--pseudoBCEMargin', type=float, default=1.0, help='margin of pseudo bce loss (default: 1.0)')
        parser.add_argument('--pseudoBCEWeight', type=float, default=1.0, help='weight of pseudo bce loss (default: 1.0)')
        parser.add_argument('--pseudoExampleNumber', type=int, default=50, help='number of pseudo positive number (default: 50)')
        parser.add_argument('--pseudoDistanceWeight', type=float, default=1.0, help='weight of pseudo Distance loss (default: 1.0)')
        parser.add_argument('--isAlphaLearnable', type=str2bool, default='True', help='whether to set alpha be learnable  (default: True)')
        parser.add_argument('--mixupEpoch', type=int, default=5, help='when to mix up (default: 5)')

    if mode == 'ILRB':
        parser.add_argument('--divide', type=int, default=-1, help='divide label matrix (default: -1)')
        # parser.add_argument('--baselineId', type=int, default=0, help='baseline model to choose corresponding (default: 0)')
        parser.add_argument('--isAlphaLearnable', type=str2bool, default='True', help='whether to set alpha be learnable  (default: True)')
        parser.add_argument('--mixupEpoch', type=int, default=5, help='when to mix up (default: 5)')

    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]    

    return args
# =============================================================================
