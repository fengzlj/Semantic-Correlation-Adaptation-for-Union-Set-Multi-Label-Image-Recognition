#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

post='SSGRL-GCN1-JoCoR-CST-COCO-VG'
printFreq=1000

mode='CST'
dataset='VG_COCO'
prob=1.0

pretrainedModel='./data/checkpoint/resnet101.pth'
resumeModel='None'
evaluate='False'

epochs=30
startEpoch=0
stepEpoch=10

batchSize=8
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512 
workers=8

co_lambda=0.1
forget_rate=0.2
exponent=1
num_gradual=15
divide=2
cross=0.0

generateLabelEpoch=5
pseudoBCEWeight=1.0
pseudoBCEMargin=0.95
pseudoDistanceWeight=0.05
pseudoExampleNumber=100

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python ./SSGRL_GCN1_JoCoR-CST-COCO-VG.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --prob ${prob} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers} \
    --divide ${divide} \
    --cross ${cross} \
    --co_lambda ${co_lambda} \
    --forget_rate ${forget_rate} \
    --exponent ${exponent} \
    --num_gradual ${num_gradual} \
    --generateLabelEpoch ${generateLabelEpoch} \
    --pseudoBCEWeight ${pseudoBCEWeight} \
    --pseudoBCEMargin ${pseudoBCEMargin} \
    --pseudoDistanceWeight ${pseudoDistanceWeight} \
    --pseudoExampleNumber ${pseudoExampleNumber}
    

